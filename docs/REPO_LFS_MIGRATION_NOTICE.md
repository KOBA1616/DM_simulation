**重要: リポジトリ履歴の LFS 移行について**

- 日時: 直近のコミットで履歴上の大きなバイナリを Git LFS に移行し、`main` ブランチを強制プッシュしました。
- 影響: リモートの `main` は履歴が書き換えられています。ローカルに同リポジトリをクローンしている協力者は、以下のいずれかの対応が必要です。

推奨対応 (簡単、安全):

1. 作業中の変更がなければ、リポジトリを再クローンする：

```powershell
cd <作業フォルダ>
git clone git@github.com:KOBA1616/DM_simulation.git
```

2. 変更がある場合（ローカルコミットがある、保存したい作業がある）:

```powershell
# 現状を保存
git branch my-work-snapshot
git push origin my-work-snapshot

# リモート main に合わせて強制的にリセット（注意: ローカルの未保存変更は失われる可能性があります）
git fetch origin
git checkout main
git reset --hard origin/main
```

代替: ローカルの変更を安全に退避してから再クローンする方法も検討してください（`git stash` / `git format-patch` 等）。

理由:
- GitHub が 100MB を超えるファイルを受け付けないため、履歴に残る大容量バイナリを LFS に移行しました。以後、大きなビルド成果物は `.gitignore` に追加しました。

問い合わせ:
- 何か問題が起きたらこの PR/Issue にコメントするか、直接連絡してください。
