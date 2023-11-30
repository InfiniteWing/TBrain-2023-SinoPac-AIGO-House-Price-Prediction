#!/bin/sh

cd ./程式碼檔案

# 需要的話，也可以手動執行以下notebook檔案

echo "Start Feature Engineering.."
jupyter nbconvert --execute --to notebook --inplace 'feature_engineering_refactoring.ipynb'
# feature_engineering_refactoring是進行程式清理後的結果，可以執行feature_engineering_org來獲得與我當初最佳提交相當的結果
# jupyter nbconvert --execute --to notebook --inplace 'feature_engineering_org.ipynb'
echo "Feature Engineering Finished！"
