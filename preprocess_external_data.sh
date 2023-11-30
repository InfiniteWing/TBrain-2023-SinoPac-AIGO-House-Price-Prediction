#!/bin/sh

cd ./程式碼檔案

# 需要的話，也可以手動執行以下notebook檔案

echo "資料前處理 - 實價登錄 - 不動產買賣"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_external_gov_data_trade.ipynb'
echo "資料前處理 - 實價登錄 - 不動產買賣年度趨勢"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_external_gov_data_by_year.ipynb'
echo "資料前處理 - 實價登錄 - 不動產租賃"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_external_gov_data_lease.ipynb'
echo "資料前處理 - 實價登錄 - 預售屋買賣"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_external_gov_data_pre_sale.ipynb'

echo "資料前處理 - 警察局地理資訊"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_police_office.ipynb'
echo "資料前處理 - 消防局地理資訊"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_fire_department.ipynb'
echo "資料前處理 - 低收入戶戶數及人數按鄉鎮市區資訊"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_low_income.ipynb'
echo "資料前處理 - 加油站地理資訊"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_gas_station.ipynb'
echo "資料前處理 - 鄉鎮土地面積及人口密度資訊"
jupyter nbconvert --execute --to notebook --inplace 'preprocess_population.ipynb'

echo "資料前處理完成！"
