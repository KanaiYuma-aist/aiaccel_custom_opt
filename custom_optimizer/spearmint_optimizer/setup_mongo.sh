#!/bin/sh

rm ./mongo_db_log_file/logfile.log*
rm -r ./mongo_db_log_file/dbfolder

mkdir ./mongo_db_log_file/dbfolder

abs_path=$(cd ./mongo_db_log_file && pwd)
log_file="/logfile.log"
dbfolder_dir="/dbfolder"

mongod --fork --logpath $abs_path$log_file --dbpath $abs_path$dbfolder_dir
