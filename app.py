"""
Jリーグ & 競馬 予測ダッシュボード
------------------------------------
起動方法(ローカル):
  streamlit run app.py

Streamlit Community Cloud で公開する場合:
  - このリポジトリをGitHubに置き、Streamlit Community Cloudから接続する
  - アプリの Settings > Secrets に以下を設定する
      APP_PASSWORD = "自分だけが知っているパスワード"
    (APP_PASSWORD を設定しない場合、パスワード確認はスキップされる = ローカル開発向け)

サイドバーの「最新データを取得して予測する」ボタンを押すと、
その場でJ1・J2両方のデータ収集(collect_jleague, Yahoo!スポーツのスクレイピング)と
予測計算(predict)が実行される。
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
CATEGORIES = collect_jleague.CATEGORIES  # {"j1": "J1", "j2": "J2"}

st.set_page_config(page_title="Jリーグ & 競馬 予測ダッシュボード", layout="wide")


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
    """J1・J2それぞれデータ収集→予測計算をまとめて実行し、CSVに保存する。"""
    counts = {}
    for category in CATEGORIES:
        results_df = collect_jleague.fetch_recent_results(category=category, last_n=None)
        results_df.to_csv(
            os.path.join(DATA_DIR, f"results_{category}_{datetime.now().date()}.csv"),
            index=False, encoding="utf-8-sig",
        )

        upcoming_df = collect_jleague.fetch_upcoming_fixtures(category=category, next_n=10)
        upcoming_df.to_csv(
            os.path.join(DATA_DIR, f"upcoming_{category}_{datetime.now().date()}.csv"),
            index=False, encoding="utf-8-sig",
        )

        form_df = collect_jleague.build_team_form(results_df)
        form_df.to_csv(
            os.path.join(DATA_DIR, f"team_form_{category}_{datetime.now().date()}.csv"),
            index=False, encoding="utf-8-sig",
        )

        predictions_df = predict.predict_fixtures(upcoming_df, form_df)
        predictions_df.to_csv(
            os.path.join(DATA_DIR, f"predictions_{category}_{datetime.now().date()}.csv"),
            index=False, encoding="utf-8-sig",
        )

        counts[category] = {"results": len(results_df), "upcoming": len(upcoming_df)}

    return counts


def _latest_file(prefix: str, category: str):
    pattern = os.path.join(DATA_DIR, f"{prefix}_{category}_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _load_latest(prefix: str, category: str):
    path = _latest_file(prefix, category)
    if path is None:
        return None
    return pd.read_csv(path)


def _recommend(row) -> pd.Series:
    """
    ホーム勝ち・引き分け・アウェイ勝ちのうち最も確率が高いものを「推奨」とし、
    1位と2位の確率差(pt)から「自信度」を判定する。
    """
    probs = {
        "ホーム勝ち": row["home_win_prob"],
        "引き分け": row["draw_prob"],
        "アウェイ勝ち": row["away_win_prob"],
    }
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_label, top_value = ranked[0]
    margin = top_value - ranked[1][1]

    if margin >= 20:
        confidence = "◎"
    elif margin >= 10:
        confidence = "○"
    else:
        confidence = "△"

    return pd.Series({"recommend": top_label, "confidence": confidence})


def render_league_tab(category: str):
    label = CATEGORIES[category]
    st.header(label)

    upcoming_df = _load_latest("upcoming", category)
    predictions_df = _load_latest("predictions", category)
    form_df = _load_latest("team_form", category)

    if upcoming_df is None or predictions_df is None:
        st.warning(
            "データがまだありません。サイドバーの「最新データを取得して予測する」を押してください。"
        )
        return

    st.subheader("今後の対戦カード・予測勝率")
    if upcoming_df.empty:
        st.info(
            "今後の対戦カードが見つかりませんでした。"
            "オフシーズン、またはデータ取得元(Yahoo!スポーツ)のページ構造が"
            "変わった可能性があります。"
        )
        return

    merged = upcoming_df.merge(
        predictions_df[["fixture_id", "home_win_prob", "draw_prob", "away_win_prob"]],
        on="fixture_id",
        how="left",
    )
    merged[["recommend", "confidence"]] = merged.apply(_recommend, axis=1)

    display_cols = [
        "date", "home_team", "away_team",
        "home_win_prob", "draw_prob", "away_win_prob",
        "recommend", "confidence",
    ]
    st.dataframe(
        merged[display_cols].rename(columns={
            "date": "日時",
            "home_team": "ホーム",
            "away_team": "アウェイ",
            "home_win_prob": "ホーム勝率(%)",
            "draw_prob": "引分率(%)",
            "away_win_prob": "アウェイ勝率(%)",
            "recommend": "推奨",
            "confidence": "自信度",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "自信度: ◎ = 1位の予測が2位を20pt以上リード / ○ = 10〜20pt差 / △ = 10pt未満の接戦。"
        "予測はあくまで直近の得失点差に基づく簡易な参考値であり、当たりを保証するものではありません。"
    )

    st.subheader(f"チーム成績({label} 今シーズン全試合)")
    if form_df is not None and not form_df.empty:
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


def render_keiba_tab():
    st.header("競馬")
    st.info("準備中です。データソースが決まり次第、対応します。")


def main():
    if not check_password():
        st.stop()

    st.title("Jリーグ & 競馬 予測ダッシュボード")

    st.sidebar.subheader("データ更新")
    if st.sidebar.button("最新データを取得して予測する"):
        with st.spinner("データ収集・予測計算 中...(J1・J2の今シーズン全試合を取得するため、数十秒かかります)"):
            try:
                counts = run_update_pipeline()
                if all(c["results"] == 0 and c["upcoming"] == 0 for c in counts.values()):
                    st.sidebar.warning(
                        "更新はできましたが、試合データが0件でした。"
                        "データ取得元(Yahoo!スポーツ)のページ構造が変わった可能性があります。"
                    )
                else:
                    st.sidebar.success("更新しました")
            except Exception as e:
                st.sidebar.error(f"更新に失敗しました: {e}")
        st.rerun()

    render_league_tab("j1")
    st.divider()
    render_league_tab("j2")
    st.divider()
    render_keiba_tab()


if __name__ == "__main__":
    main()
