"""
Jリーグ 試合結果予測スクリプト
------------------------------------
collect_jleague.py が出力した「消化済み試合結果(results)」と「今後の対戦カード(upcoming)」
の最新CSVを読み込み、各対戦カードの勝敗予測(勝率%)とチームのELOレーティングを計算して
CSVに保存する。

採用している手法(いずれもサッカー予想で伝統的に使われる統計手法):

1. ポアソン分布モデル(勝率計算のベース)
   各チームの「平均得点力」「平均失点力」をリーグ平均との比率で表し、
   対戦カードごとに期待得点(平均何点入りそうか)を算出。
   その期待得点をポアソン分布に当てはめて、有り得る全スコアの組み合わせから
   ホーム勝ち/引き分け/アウェイ勝ちの確率を積み上げて計算する。

2. 直近重視の重み付け
   得点力・失点力を算出する際、節が古い試合ほど重みを下げる(指数減衰)。
   シーズン全体の平均だけでなく「今の調子」をより反映できるようにしている。

3. ELOレーティングとの統合
   対戦結果を古い順に処理し、勝てば相手の強さに応じてレーティングが上がり、
   負ければ下がる仕組み(チェスのレーティングと同じ考え方)。
   このELOの実力差を、ポアソンモデルの期待得点に補正として掛け合わせている。
   「どんな相手に勝ったか」を、得点力とは別の角度から反映する狙い。

4. Dixon-Coles補正(引き分け精度の補正)
   実際のサッカーでは、単純な(ホーム・アウェイの得点が完全に独立という前提の)
   ポアソン分布が予測するより、「0-0」「1-1」のような低スコアの引き分けが
   やや多く発生することが知られている(Dixon & Coles, 1997)。
   この既知の補正式を、低スコアの組み合わせにだけ適用している。

使い方:
  python scripts/predict.py
"""

import glob
import math
import os
from collections import defaultdict
from datetime import datetime

import pandas as pd

import collect_jleague

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jleague")
CATEGORIES = collect_jleague.CATEGORIES  # {"j1": "J1", "j2": "J2"}

# ===== ELOレーティングの設定 =====
ELO_INITIAL = 1500       # 全チームの初期レーティング
ELO_K = 20                # 1試合あたりのレーティング変動の大きさ(大きいほど直近の結果を重視する)
ELO_HOME_ADVANTAGE = 100  # ホームチームに上乗せする補正値

# ===== ポアソンモデルの設定 =====
POISSON_MAX_GOALS = 8     # 何点まで想定してスコアの組み合わせを計算するか(通常はこれで十分)
POISSON_PRIOR_GAMES = 4   # 試合数が少ないチームの数値をリーグ平均に寄せるための仮想試合数
                           # (シーズン序盤は数試合しかないため、そのままだと極端な予測になりやすい)
RECENCY_DECAY = 0.90       # 1節古くなるごとに重みをこの倍率にする(直近を重視するほど1.0から離す)
ELO_BLEND_WEIGHT = 0.5    # ポアソンの期待得点にELOの実力差をどれだけ反映するか(0=反映しない, 1=フル反映)
DIXON_COLES_RHO = -0.13   # 低スコアの出やすさ補正値。Dixon & Coles(1997)で一般的に使われる値


def _latest_file(prefix: str, category: str) -> str:
    pattern = os.path.join(DATA_DIR, f"{prefix}_{category}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"{pattern} が見つかりません。先に collect_jleague.py を実行してください。"
        )
    return files[-1]


def _add_round_no(results_df: pd.DataFrame) -> pd.DataFrame:
    """"round"列(例: "第3節")から節番号を数値として取り出した列を追加する。"""
    df = results_df.copy()
    df["round_no"] = df["round"].fillna("").str.extract(r"(\d+)")
    df = df.dropna(subset=["round_no"])
    df["round_no"] = df["round_no"].astype(int)
    return df


def compute_elo_ratings(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    消化済み試合結果を節(ラウンド)の古い順に処理し、チームごとのELOレーティングを算出する。
    同じ節の試合は同時に行われたものとして扱い、節が終わってからまとめてレーティングを更新する。
    """
    columns = ["team", "elo"]
    if results_df.empty:
        return pd.DataFrame(columns=columns)

    df = _add_round_no(results_df)
    if df.empty:
        return pd.DataFrame(columns=columns)

    ratings = defaultdict(lambda: ELO_INITIAL)

    for _, round_df in df.sort_values("round_no").groupby("round_no"):
        pending_updates = defaultdict(float)
        for _, row in round_df.iterrows():
            home, away = row["home_team"], row["away_team"]
            home_goals, away_goals = row["home_goals"], row["away_goals"]

            expected_home = 1 / (1 + 10 ** (((ratings[away]) - (ratings[home] + ELO_HOME_ADVANTAGE)) / 400))

            if home_goals > away_goals:
                actual_home = 1.0
            elif home_goals < away_goals:
                actual_home = 0.0
            else:
                actual_home = 0.5

            delta = ELO_K * (actual_home - expected_home)
            pending_updates[home] += delta
            pending_updates[away] -= delta

        for team, delta in pending_updates.items():
            ratings[team] += delta

    records = [{"team": team, "elo": round(rating)} for team, rating in ratings.items()]
    return pd.DataFrame(records, columns=columns).sort_values("elo", ascending=False)


def _shrunk_rate(weighted_goals_sum: float, weighted_games: float, league_avg_rate: float) -> float:
    """
    観測された(重み付き)平均に、リーグ平均を「仮想的にPOISSON_PRIOR_GAMES試合分」混ぜて寄せる。
    試合数(の重み合計)が少ないほどリーグ平均寄りになり、増えるほど実際の数値に近づいていく。
    """
    return (weighted_goals_sum + POISSON_PRIOR_GAMES * league_avg_rate) / (weighted_games + POISSON_PRIOR_GAMES)


def _poisson_params(results_df: pd.DataFrame):
    """
    リーグ平均得点と、チームごと(ホーム/アウェイ別)の攻撃力・守備力の係数を算出する。
    係数は「リーグ平均を1.0としたときの倍率」で表す(1.2なら平均より2割得点力が高い、など)。
    節が古い試合ほど重みを下げて(RECENCY_DECAY)、直近の調子を反映しやすくしている。
    """
    df = _add_round_no(results_df)
    if df.empty:
        return None

    current_round = df["round_no"].max()
    df["_weight"] = RECENCY_DECAY ** (current_round - df["round_no"])

    league_avg_home_goals = (df["home_goals"] * df["_weight"]).sum() / df["_weight"].sum()
    league_avg_away_goals = (df["away_goals"] * df["_weight"]).sum() / df["_weight"].sum()
    if not league_avg_home_goals or not league_avg_away_goals:
        return None

    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    team_stats = {}
    for team in teams:
        home_games = df[df["home_team"] == team]
        away_games = df[df["away_team"] == team]

        home_attack_rate = _shrunk_rate(
            (home_games["home_goals"] * home_games["_weight"]).sum(), home_games["_weight"].sum(), league_avg_home_goals
        )
        home_defense_rate = _shrunk_rate(
            (home_games["away_goals"] * home_games["_weight"]).sum(), home_games["_weight"].sum(), league_avg_away_goals
        )
        away_attack_rate = _shrunk_rate(
            (away_games["away_goals"] * away_games["_weight"]).sum(), away_games["_weight"].sum(), league_avg_away_goals
        )
        away_defense_rate = _shrunk_rate(
            (away_games["home_goals"] * away_games["_weight"]).sum(), away_games["_weight"].sum(), league_avg_home_goals
        )

        team_stats[team] = {
            "home_attack": home_attack_rate / league_avg_home_goals,
            "home_defense": home_defense_rate / league_avg_away_goals,
            "away_attack": away_attack_rate / league_avg_away_goals,
            "away_defense": away_defense_rate / league_avg_home_goals,
        }

    return {
        "league_avg_home_goals": league_avg_home_goals,
        "league_avg_away_goals": league_avg_away_goals,
        "team_stats": team_stats,
    }


def _expected_goals(params: dict, elo_ratings: dict, home_team: str, away_team: str):
    """
    指定した対戦カードの期待得点(平均何点入りそうか)を、ホーム・アウェイそれぞれ算出する。
    得点力・失点力(ポアソン)をベースに、ELOレーティングの実力差で補正する。
    """
    default_stats = {"home_attack": 1.0, "home_defense": 1.0, "away_attack": 1.0, "away_defense": 1.0}
    home_stats = params["team_stats"].get(home_team, default_stats)
    away_stats = params["team_stats"].get(away_team, default_stats)

    lambda_home = home_stats["home_attack"] * away_stats["away_defense"] * params["league_avg_home_goals"]
    lambda_away = away_stats["away_attack"] * home_stats["home_defense"] * params["league_avg_away_goals"]

    elo_home = elo_ratings.get(home_team, ELO_INITIAL)
    elo_away = elo_ratings.get(away_team, ELO_INITIAL)
    # ELOの差を、期待得点への倍率に変換する(差が無ければ倍率1.0)。
    # ホーム側にかける分とアウェイ側で割る分に半分ずつ配分し、全体の得点量が偏りすぎないようにする。
    elo_adjustment = 10 ** ((elo_home - elo_away) * ELO_BLEND_WEIGHT / 800)
    lambda_home *= elo_adjustment
    lambda_away /= elo_adjustment

    return lambda_home, lambda_away


def _poisson_pmf(k: int, lam: float) -> float:
    """ポアソン分布: 平均lam回起きる事象が、ちょうどk回起きる確率。"""
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dixon_coles_tau(h: int, a: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """
    低スコア(0-0, 1-0, 0-1, 1-1)の組み合わせにだけ適用する補正係数。
    それ以外のスコアは補正なし(1.0)。
    """
    if h == 0 and a == 0:
        return 1 - (lambda_home * lambda_away * rho)
    if h == 1 and a == 0:
        return 1 + (lambda_home * rho)
    if h == 0 and a == 1:
        return 1 + (lambda_away * rho)
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def _match_probabilities(lambda_home: float, lambda_away: float, max_goals: int = POISSON_MAX_GOALS):
    """
    期待得点(lambda_home, lambda_away)から、有り得るスコアの組み合わせを全て足し合わせて
    ホーム勝ち/引き分け/アウェイ勝ちの確率を計算する。Dixon-Coles補正込み。
    """
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        p_h = _poisson_pmf(h, lambda_home)
        for a in range(max_goals + 1):
            p = p_h * _poisson_pmf(a, lambda_away) * _dixon_coles_tau(h, a, lambda_home, lambda_away, DIXON_COLES_RHO)
            p = max(p, 0.0)  # 補正で理論上わずかに負になり得る場合の保険
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

    total = home_win + draw + away_win  # max_goalsを超える裾野や補正分の誤差を正規化して吸収する
    return home_win / total, draw / total, away_win / total


def predict_fixtures(upcoming_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    params = _poisson_params(results_df)
    elo_df = compute_elo_ratings(results_df)
    elo_ratings = dict(zip(elo_df["team"], elo_df["elo"])) if not elo_df.empty else {}

    rows = []
    for _, fixture in upcoming_df.iterrows():
        if params is None:
            home_win_prob, draw_prob, away_win_prob = 33.3, 33.4, 33.3  # 実績データがまだ無い場合の暫定値
        else:
            lambda_home, lambda_away = _expected_goals(params, elo_ratings, fixture["home_team"], fixture["away_team"])
            home_win, draw, away_win = _match_probabilities(lambda_home, lambda_away)
            home_win_prob, draw_prob, away_win_prob = round(home_win * 100, 1), round(draw * 100, 1), round(away_win * 100, 1)

        rows.append({
            "fixture_id": fixture["fixture_id"],
            "date": fixture["date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "home_win_prob": home_win_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_win_prob,
        })

    columns = ["fixture_id", "date", "home_team", "away_team", "home_win_prob", "draw_prob", "away_win_prob"]
    return pd.DataFrame(rows, columns=columns)


def main():
    for category, label in CATEGORIES.items():
        print(f"\n===== {label} =====")
        upcoming_df = pd.read_csv(_latest_file("upcoming", category))
        results_df = pd.read_csv(_latest_file("results", category))

        predictions_df = predict_fixtures(upcoming_df, results_df)
        predictions_path = os.path.join(DATA_DIR, f"predictions_{category}_{datetime.now().date()}.csv")
        predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        print(f"保存しました: {predictions_path}")
        print(predictions_df)

        elo_df = compute_elo_ratings(results_df)
        elo_path = os.path.join(DATA_DIR, f"elo_{category}_{datetime.now().date()}.csv")
        elo_df.to_csv(elo_path, index=False, encoding="utf-8-sig")
        print(f"保存しました: {elo_path}")
        print(elo_df.head())


if __name__ == "__main__":
    main()
