#!/bin/sh

# 消防局地點資料
wget -O "外部資料集/fire_department.csv" "https://data.moi.gov.tw/MoiOD/System/DownloadFile.aspx?DATA=C38B7AC2-E7F3-4DD5-A3F3-88E623B55924"

# 警察局地點資料
wget -O "外部資料集/police_office.ods" "https://www.npa.gov.tw/ch/app/data/doc?module=liaison&detailNo=1174640233954676736&type=s"

# 台糖加油站位置
wget -O "外部資料集/gas_station_1.csv" "https://www.taisugar.com.tw/Upload/UserFiles/%E5%8F%B0%E7%B3%96%E5%8A%A0%E6%B2%B9%E7%AB%99%E8%B3%87%E8%A8%8A1020508.csv"

# 中油加油站位置
wget -O "外部資料集/gas_station_2.xml" "https://vipmbr.cpc.com.tw/CPCSTN/STNWebService.asmx/getStationInfo_XML"

# 鄉鎮土地面積及人口密度(97)
# 請手動下載，或者使用我當初下載的版本
# https://www.ris.gov.tw/app/portal/346


# 低收入戶戶數及人數按鄉鎮市區別分
wget -O "外部資料集/1.1.2低收入戶戶數及人數按鄉鎮市區別分112Q2.ods" "https://www.mohw.gov.tw/dl-27853-3b9c8bb7-dc12-49f3-8e61-cb2dd589003f.html"


# 內政部不動產成交案件實際資訊資料

mkdir -p "外部資料集/實價登錄/2020Q1/"
wget -O "外部資料集/實價登錄/2020Q1/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=109S1&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2020Q1/lvr_landcsv.zip" -d "外部資料集/實價登錄/2020Q1/"


mkdir -p "外部資料集/實價登錄/2020Q2/"
wget -O "外部資料集/實價登錄/2020Q2/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=109S2&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2020Q2/lvr_landcsv.zip" -d "外部資料集/實價登錄/2020Q2/"


mkdir -p "外部資料集/實價登錄/2020Q3/"
wget -O "外部資料集/實價登錄/2020Q3/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=109S3&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2020Q3/lvr_landcsv.zip" -d "外部資料集/實價登錄/2020Q3/"


mkdir -p "外部資料集/實價登錄/2020Q4/"
wget -O "外部資料集/實價登錄/2020Q4/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=109S4&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2020Q4/lvr_landcsv.zip" -d "外部資料集/實價登錄/2020Q4/"


mkdir -p "外部資料集/實價登錄/2021Q1/"
wget -O "外部資料集/實價登錄/2021Q1/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=110S1&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2021Q1/lvr_landcsv.zip" -d "外部資料集/實價登錄/2021Q1/"


mkdir -p "外部資料集/實價登錄/2021Q2/"
wget -O "外部資料集/實價登錄/2021Q2/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=110S2&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2021Q2/lvr_landcsv.zip" -d "外部資料集/實價登錄/2021Q2/"


mkdir -p "外部資料集/實價登錄/2021Q3/"
wget -O "外部資料集/實價登錄/2021Q3/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=110S3&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2021Q3/lvr_landcsv.zip" -d "外部資料集/實價登錄/2021Q3/"


mkdir -p "外部資料集/實價登錄/2021Q4/"
wget -O "外部資料集/實價登錄/2021Q4/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=110S4&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2021Q4/lvr_landcsv.zip" -d "外部資料集/實價登錄/2021Q4/"


mkdir -p "外部資料集/實價登錄/2022Q1/"
wget -O "外部資料集/實價登錄/2022Q1/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=111S1&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2022Q1/lvr_landcsv.zip" -d "外部資料集/實價登錄/2022Q1/"


mkdir -p "外部資料集/實價登錄/2022Q2/"
wget -O "外部資料集/實價登錄/2022Q2/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=111S2&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2022Q2/lvr_landcsv.zip" -d "外部資料集/實價登錄/2022Q2/"


mkdir -p "外部資料集/實價登錄/2022Q3/"
wget -O "外部資料集/實價登錄/2022Q3/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=111S3&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2022Q3/lvr_landcsv.zip" -d "外部資料集/實價登錄/2022Q3/"


mkdir -p "外部資料集/實價登錄/2022Q4/"
wget -O "外部資料集/實價登錄/2022Q4/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=111S4&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2022Q4/lvr_landcsv.zip" -d "外部資料集/實價登錄/2022Q4/"


mkdir -p "外部資料集/實價登錄/2023Q1/"
wget -O "外部資料集/實價登錄/2023Q1/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=112S1&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2023Q1/lvr_landcsv.zip" -d "外部資料集/實價登錄/2023Q1/"


mkdir -p "外部資料集/實價登錄/2023Q2/"
wget -O "外部資料集/實價登錄/2023Q2/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=112S2&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2023Q2/lvr_landcsv.zip" -d "外部資料集/實價登錄/2023Q2/"


mkdir -p "外部資料集/實價登錄/2023Q3/"
wget -O "外部資料集/實價登錄/2023Q3/lvr_landcsv.zip" "https://plvr.land.moi.gov.tw//DownloadSeason?season=112S3&type=zip&fileName=lvr_landcsv.zip"
unzip "外部資料集/實價登錄/2023Q3/lvr_landcsv.zip" -d "外部資料集/實價登錄/2023Q3/"


md5sum 外部資料集/*

echo "Finish Downloading External Data"