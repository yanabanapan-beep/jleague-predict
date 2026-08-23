"""
Jリーグ & 競馬 予測ダッシュボード
------------------------------------
起動方法(ローカル):
  streamlit run app.py

Streamlit Community Cloud で公開する場合:
  - このリポジトリをGitHubに置き、Streamlit Community Cloudから接続する
  - アプリの Settings > Secrets に以下を設定する
      API_FOOTBALL_KEY = "取得したAPI-FootballのAPIキー"
      APP_PASSWORD = "自分だけが知っているパスワード"
    (APP_PASSWORD を設定しない場合、パスワード確認はスキップされる = ローカル開発向け)

サイドバーの「最新データを取得して予測する」ボタンを押すと、
その場でデータ収集(collect_jleague)と予測計算(predict)が実行される。
"""

import glob
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import collect_jleague  # noqa: E402
import predict  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "jleague")
SEASON = 2026  # 必要に応じて変更

st.set_page_config(page_title="Jリーグ & 競馬 予測ダッシュボード", layout="wide")


def _sync_api_key_from_secrets():
    """Streamlit CloudのSecretsに設定されたAPIキーを環境変数へ反映する。"""
    api_key = st.secrets.get("API_FOOTBALL_KEY")
    if api_key:
        os.environ["API_FOOTBALL_KEY"] = api_key


def check_password() -> bool:
    """
    Secretsに APP_PASSWORD が設定されている場合のみパスワード確認を行う。
    (自分だけが見られればよい、という限定公開の要件に対する簡易的な認証)
    """
    app_password = st.secrets.get("APP_PASSWORD")
    if not app_password:
        return True  # パスワード未設定(ローカル開発など)はスキップ

    if st.session_state.get("password_correct"):
        return True

    entered = st.text_input("パスワード", type="password")
    if entered:
        if entered == app_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    return False


def run_update_pipeline():
    """データ収集→予測計算をまとめて実行し、CSVに保存する。"""
    results_df = collect_jleague.fetch_recent_results(season=SEASON, last_n=30)
    results_df.to_csv(
        os.path.join(DATA_DIR, f"results_{datetime.now().date()}.csv"),
        index=False, encoding="utf-8-sig",
    )

    upcoming_df = collect_jleague.fetch_upcoming_fixtures(season=SEASON, next_n=10)
    upcoming_df.to_csv(
        os.path.join(DATA_DIR, f"upcoming_{datetime.now().date()}.csv"),
        index=False, encoding="utf-8-sig",
    )

    form_df = collect_jleague.build_team_form(results_df)
    form_df.to_csv(
        os.path.join(DATA_DIR, f"team_form_{datetime.now().date()}.csv"),
        index=False, encoding="utf-8-sig",
    )

    injuries_df = collect_jleague.fetch_injuries(season=SEASON)
    injuries_df.to_csv(
        os.path.join(DATA_DIR, f"injuries_{datetime.now().date()}.csv"),
        index=False, encoding="utf-8-sig",
    )

    predictions_df = predict.predict_fixtures(upcoming_df, form_df)
    predictions_df.to_csv(
        os.path.join(DATA_DIR, f"predictions_{datetime.now().date()}.csv"),
        index=False, encoding="utf-8-sig",
    )

    return {
        "results": len(results_df),
        "upcoming": len(upcoming_df),
        "injuries": len(injuries_df),
    }


def _latest_file(prefix: str):
    pattern = os.path.join(DATA_DIR, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _load_latest(prefix: str):
    path = _latest_file(prefix)
    if path is None:
        return None
    return pd.read_csv(path)


def render_jleague_tab():
    st.header("Jリーグ")

    upcoming_df = _load_latest("upcoming")
    predictions_df = _load_latest("predictions")
    form_df = _load_latest("team_form")
    injuries_df = _load_latest("injuries")

    if upcoming_df is None or predictions_df is None:
        st.warning(
            "データがまだありません。先に以下を実行してください。\n\n"
            "1. `python scripts/collect_jleague.py`\n"
            "2. `python scripts/predict.py`"
        )
        return

    st.subheader("今後の対戦カード・予測勝率")
    if upcoming_df.empty:
        st.info(
            "今後の対戦カードが見つかりませんでした。"
            "オフシーズン、またはAPI-Footballの契約プランがJリーグ(J1)のデータに"
            "対応していない可能性があります。"
        )
        return
    merged = upcoming_df.merge(
        predictions_df[["fixture_id", "home_win_prob", "draw_prob", "away_win_prob"]],
        on="fixture_id",
        how="left",
    )
    if injuries_df is not None and not injuries_df.empty:
        injury_counts = injuries_df.groupby("team")["player"].count()
        merged["home_injury_alert"] = merged["home_team"].map(injury_counts).fillna(0).astype(int)
        merged["away_injury_alert"] = merged["away_team"].map(injury_counts).fillna(0).astype(int)
        merged["home_team"] = merged.apply(
            lambda r: f"⚠ {r['home_team']}" if r["home_injury_alert"] > 0 else r["home_team"],
            axis=1,
        )
        merged["away_team"] = merged.apply(
            lambda r: f"⚠ {r['away_team']}" if r["away_injury_alert"] > 0 else r["away_team"],
            axis=1,
        )
        merged = merged.drop(columns=["home_injury_alert", "away_injury_alert"])

    display_cols = ["date", "home_team", "away_team", "home_win_prob", "draw_prob", "away_win_prob"]
    st.dataframe(
        merged[display_cols].rename(columns={
            "date": "日時",
            "home_team": "ホーム",
            "away_team": "アウェイ",
            "home_win_prob": "ホーム勝率(%)",
            "draw_prob": "引分率(%)",
            "away_win_prob": "アウェイ勝率(%)",
        }),
        use_container_width=True,
        hide_index=True,
    )
    if injuries_df is not None and not injuries_df.empty:
        st.caption("⚠ = 出場停止・負傷情報あり(下の一覧を参照)")

    st.subheader("チーム成績(直近)")
    if form_df is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                form_df.rename(columns={
                    "team": "チーム",
                    "games_played": "試合数",
                    "goals_for": "得点",
                    "goals_against": "失点",
                    "goal_diff": "得失点差",
                    "wins": "勝利数",
                }),
                use_container_width=True,
                hide_index=True,
            )
        with col2:
            st.bar_chart(form_df.set_index("team")["goal_diff"])

    if injuries_df is not None and not injuries_df.empty:
        st.subheader("出場停止・負傷情報")
        st.dataframe(
            injuries_df.rename(columns={
                "team": "チーム",
                "player": "選手",
                "type": "区分",
                "reason": "理由",
                "fixture_date": "対象試合日",
            }),
            use_container_width=True,
            hide_index=True,
        )


def render_keiba_tab():
    st.header("競馬")
    st.info("準備中です。データソースが決まり次第、対応します。")


def main():
    _sync_api_key_from_secrets()

    if not check_password():
        st.stop()

    st.title("Jリーグ & 競馬 予測ダッシュボード")

    st.sidebar.subheader("データ更新")
    if st.sidebar.button("最新データを取得して予測する"):
        with st.spinner("データ収集・予測計算 中..."):
            try:
                counts = run_update_pipeline()
                if counts["results"] == 0 and counts["upcoming"] == 0:
                    st.sidebar.warning(
                        "更新はできましたが、試合データが0件でした。"
                        "オフシーズン、またはAPI-Footballの契約プランがJリーグ(J1)のデータに"
                        "対応していない可能性があります。"
                    )
                else:
                    st.sidebar.success("更新しました")
            except Exception as e:
                st.sidebar.error(f"更新に失敗しました: {e}")
        st.rerun()

    tab = st.sidebar.radio("表示するデータ", ["Jリーグ", "競馬"])

    if tab == "Jリーグ":
        render_jleague_tab()
    else:
        render_keiba_tab()


if __name__ == "__main__":
    main()
