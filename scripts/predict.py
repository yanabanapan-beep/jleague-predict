"""
Jリーグ 試合結果予測スクリプト
------------------------------------
collect_jleague.py が出力した「チーム成績(team_form)」と「今後の対戦カード(upcoming)」
の最新CSVを読み込み、各対戦カードの勝敗予測(勝率%)を計算してCSVに保存する。

予測ロジックについて:
  - 今はまだ過去の「予測 vs 実際の結果」データが十分に無いため、
    LightGBMなどの機械学習モデルを学習させることができない。
  - そのため、まずは「チームの直近の得失点差」をベースにしたシンプルな計算式
    (ソフトマックス: ロジスティック回帰を3択に拡張したもの)で予測を出す。
  - 十分な実績データが溜まったら、本格的な学習モデルに置き換える想定。

使い方:
  python scripts/predict.py
"""

import glob
import math
import os
from datetime import datetime

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jleague")

# ホームチームの有利さ(得失点差に上乗せする補正値)。値が大きいほどホーム有利を強く見る。
HOME_ADVANTAGE = 0.3
# 引き分けの起こりやすさのベース値。Jリーグの引き分け率(実績上おおよそ2割前後)を踏まえた目安。
DRAW_BASELINE = 0.9
# 得失点差の影響の強さ。値が大きいほど成績差がそのまま勝率差に反映されやすくなる。
STRENGTH_SCALE = 0.6


def _latest_file(prefix: str) -> str:
    pattern = os.path.join(DATA_DIR, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"{pattern} が見つかりません。先に collect_jleague.py を実行してください。"
        )
    return files[-1]


def _team_strength(form_df: pd.DataFrame, team: str) -> float:
    row = form_df[form_df["team"] == team]
    if row.empty or row.iloc[0]["games_played"] == 0:
        return 0.0
    goal_diff = row.iloc[0]["goal_diff"]
    games_played = row.iloc[0]["games_played"]
    return goal_diff / games_played


def predict_fixtures(upcoming_df: pd.DataFrame, form_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, fixture in upcoming_df.iterrows():
        home_strength = _team_strength(form_df, fixture["home_team"])
        away_strength = _team_strength(form_df, fixture["away_team"])

        score_home = (home_strength + HOME_ADVANTAGE) * STRENGTH_SCALE
        score_away = away_strength * STRENGTH_SCALE
        score_draw = DRAW_BASELINE

        exp_home = math.exp(score_home)
        exp_away = math.exp(score_away)
        exp_draw = math.exp(score_draw)
        total = exp_home + exp_away + exp_draw

        rows.append({
            "fixture_id": fixture["fixture_id"],
            "date": fixture["date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "home_win_prob": round(exp_home / total * 100, 1),
            "draw_prob": round(exp_draw / total * 100, 1),
            "away_win_prob": round(exp_away / total * 100, 1),
        })

    return pd.DataFrame(rows)


def main():
    upcoming_path = _latest_file("upcoming")
    form_path = _latest_file("team_form")

    upcoming_df = pd.read_csv(upcoming_path)
    form_df = pd.read_csv(form_path)

    predictions_df = predict_fixtures(upcoming_df, form_df)

    predictions_path = os.path.join(DATA_DIR, f"predictions_{datetime.now().date()}.csv")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    print(f"保存しました: {predictions_path}")
    print("\n=== 予測結果 ===")
    print(predictions_df)


if __name__ == "__main__":
    main()
