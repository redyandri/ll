from __future__ import division
from numpy import median
from tool.csv_feature_generator import *
from KNNClassifier import KNNClassifier
from my_tool import *
from sklearn.model_selection import train_test_split
from victorinox import victorinox
# new_size=[224,224]
#
# uw3_zones_noredundant_allclass="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_allclass"
# uw3_zones_noredundant_allclass_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_allclass_csv.csv"
# uw3_zones_noredundant_hifreq="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq"
# uw3_zones_noredundant_hifreq_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224"
# uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_relocated_blackpadded_224="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_reshaped_relocated_blackpadded_224"
# uw3_zones_noredundant_hifreq_reshaped_relocated_blackpadded_224_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_reshaped_relocated_blackpadded_224_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224"
# uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_relocated_224_mini="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_reshaped_relocated_224_mini"
# uw3_zones_noredundant_hifreq_reshaped_relocated_224_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_reshaped_relocated_224_mini_csv.csv"
# uw3_zones_noredundant_allclass_histogram_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_allclass_histogram_csv.csv"
# uw3_zones_noredundant_hifreq_histogram_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_zones_noredundant_hifreq_histogram_csv.csv"
# uw3_zones_tt_test_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_test_split/uw3_zones_tt_test_csv.csv"
# uw3_zones_tt_train_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_test_split/uw3_zones_tt_train_csv.csv"
# uw3_zones_tt_test_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_test_split/uw3_zones_tt_test_mini_csv.csv"
# uw3_zones_tt_train_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_test_split/uw3_zones_tt_train_mini_csv.csv"
# uw3_zones_tvt_test_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_tvt_test_csv.csv"
# uw3_zones_tvt_train_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_tvt_train_csv.csv"
# uw3_zones_tvt_val_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_tvt_val_csv.csv"
# uw3_zones_tvt_test_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_tvt_test_mini_csv.csv"
# uw3_zones_tvt_train_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_tvt_train_mini_csv.csv"
# uw3_zones_tvt_val_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_tvt_val_mini_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224_train_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224_train_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224_val_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224_val_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224_test_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_noredundant_hifreq_reshaped_relocated_whitepadded_224_test_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224_train_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224_train_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224_val_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224_val_csv.csv"
# uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224_test_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/uw3_train_val_test_split/uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224_tets_csv.csv"
#
# ocrd_zones_allclass="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_allclass"
# ocrd_zones_allclass_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_allclass_csv.csv"
# ocrd_zones_hifreq="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq"
# ocrd_zones_hifreq_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_csv.csv"
# ocrd_zones_hifreq_reshaped_relocated_whitepadded_224="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_reshaped_relocated_whitepadded_224"
# ocrd_zones_hifreq_reshaped_relocated_whitepadded_224_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_reshaped_relocated_whitepadded_224_csv.csv"
# ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224"
# ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224_csv.csv"
#
# ocrd_zones_hifreq_reshaped_relocated_blackpadded_224="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_reshaped_relocated_blackpadded_224"
# ocrd_zones_hifreq_reshaped_relocated_blackpadded_224_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_reshaped_relocated_blackpadded_224_csv.csv"
# ocrd_zones_hifreq_reshaped_relocated_224_mini="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_reshaped_relocated_224_mini"
# ocrd_zones_hifreq_reshaped_relocated_224_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_reshaped_relocated_224_mini_csv.csv"
# ocrd_tvt_train="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_val_test_split/ocrd_tvt_train.csv"
# ocrd_tvt_val="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_val_test_split/ocrd_tvt_val.csv"
# ocrd_tvt_test="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_val_test_split/ocrd_tvt_test.csv"
# ocrd_tvt_train_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_val_test_split/ocrd_tvt_train_mini_csv.csv"
# ocrd_tvt_val_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_val_test_split/ocrd_tvt_val_mini_csv.csv"
# ocrd_tvt_test_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_val_test_split/ocrd_tvt_test_mini_csv.csv"
# ocrd_tt_train="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_tt_train.csv"
# ocrd_tt_test="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_tt_test.csv"
# ocrd_tt_train_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_tt_train_mini_csv.csv"
# ocrd_tt_test_mini_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_tt_test_mini_csv.csv"
# ocrd_zones_allclass_histogram_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_allclass_histogram_csv.csv"
# ocrd_zones_hifreq_histogram_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_zones_hifreq_histogram_csv.csv"
# ocrd_zones_hifreq_reshaped_relocated_whitepadded_224_train_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_zones_hifreq_reshaped_relocated_whitepadded_224_train_csv.csv"
# ocrd_zones_hifreq_reshaped_relocated_whitepadded_224_val_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_zones_hifreq_reshaped_relocated_whitepadded_224_val_csv.csv"
# ocrd_zones_hifreq_reshaped_relocated_whitepadded_224_test_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_zones_hifreq_reshaped_relocated_whitepadded_224_test_csv.csv"
# ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224_train_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224_train_csv.csv"
# ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224_val_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224_val_csv.csv"
# ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224_test_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/ocrd_train_test_split/ocrd_zones_hifreq_reshaped_unrelocated_whitepadded_224_test_csv.csv"
#
#
#
# # hase_path="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/hase/hase.csv"
# # path="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/dataset/obsolete_rlbwxh.csv"
# # path1="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/dataset/histograms/ccxyh.csv"
# # path2="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/dataset/histograms/orisize_rlbwxyh_hifreq_class.csv"
# # knn=KNNClassifier()
# # knn.classifyKNN_by_rlbwxyh(path2)
# # knn.classifyKNN_by_weighted_distance([path1,path2])
#
g=csv_feature_generator()
#uw3_ori="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/original"
uw3_aug="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented"
uw3_aug_rlbwxh_result="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/uw3_aug_rlbwxh_result.csv"
#ocrd_ori="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/original"
ocrd_aug="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented"
ocrd_aug_rlbwxh_result="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented_rlbwxh_result.csv"
# # img_folder="/home/andri/Documents/s2/5/master_arbeit/dataset/UWIII/ZONE_NO_REDUNDANCY"
# # csv_folder="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/dataset/histograms/orisize_rlbwxyh_full_class.csv"
# # # #img_folder="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/hase/A00A"
# # # #csv_folder="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/hase"
# g.convert_images_to_rlbwxyh_csv(img_folder=ocrd_aug,#uw3_aug,
#                                 csv_path=ocrd_aug_rlbwxh_result,#uw3_aug_rlbwxh_result,
#                                 text_resume_idx=326900,#181700,#0,
#                                 save_batch=100)
#
#
# # ocr_d_folder="/home/andri/Documents/s2/5/master_arbeit/dataset/historical/binarized_zones"
# # ocr_d_rlbwxyh_file="/home/andri/Documents/s2/5/master_arbeit/app/logical_labeling/dataset/histograms/ocrd_orisize_rlbwxyh_16classes.csv"
# # #rlbwxyh=RLXYMSH()
# # #g=csv_feature_generator()
# ocrd_histogram_dataset_file="/home/andri/Documents/s2/5/master_arbeit/dataset/historical/histogram_dataset.csv"
# uw3_hifreq_histogram_dataset_file="/home/andri/Documents/s2/5/master_arbeit/app/logical_labeling/dataset/histograms/orisize_rlbwxyh_hifreq_class.csv"

uw3_hifreq="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/uw3_zones_noredundant_hifreq_reshaped_unrelocated_whitepadded_224"
hifreq="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/unaugmented"
uw3_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_csv.csv"
uw3_result_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_ori_result.csv"
ocrd_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_csv.csv"
uw3_clustered="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_CLUSTERED_csv.csv"
ocrd_clustered="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_CLUSTERED_csv.csv"
uw3_clustered_balanced="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_CLUSTERED_BALANCED_csv.csv"
ocrd_clustered_balanced="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_CLUSTERED_BALANCEDcsv.csv"
bal="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/balanced.csv"
ocrd="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented"
uw3="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented"
knn=KNNClassifier()
uw3_aug_rlbwxyh="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/uw3_aug_rlbwxyh.csv"
aug_res="/home/andri/Documents/s2/5/master_arbeit/app/logical_labeling/test_model/res_aug_rlbwxh.csv"
uw3_ori_hist_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_ori_hist_train.csv"
uw3_ori_hist_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_ori_hist_test.csv"
uw3_ori_result_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_ori_result.csv"
ocrd_ori_hist="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_csv.csv"
ocrd_result_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_result.csv"
tool=my_tool()
# class_dict=tool.get_classname_dictionary(dataset_path=ocrd)
# class_dict={v:k for k,v in class_dict.items()}
# knn.classifyKNN_by_rlbwxyh(CSVFilePath=ocrd_ori_hist,#uw3_aug_rlbwxyh,#
#                            class_dict=class_dict,n_splits=5,
#                            result_csv=ocrd_result_csv,#aug_res,
#                            resume_test_idx=0) #make 0.2 for test, crossval
# from victorinox import victorinox
# uw3_dct_csv="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/uw3_dictionary.csv"
# vic=victorinox()
# uw3_dict=vic.convert_csv_to_dictionary(uw3_dct_csv)
# knn.classifyKNN_by_rlbwxyh_by_train_val_csv(train_csv=uw3_ori_hist_train_csv,#uw3_aug_rlbwxyh,#
#                            test_csv=uw3_ori_hist_test_csv,
#                            class_dict=uw3_dict,
#                            result_csv=uw3_result_csv,#aug_res,
#                           resume_test_idx=0) #make 0.2 for test, crossval

uw3_dct_csv="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/uw3_dictionary.csv"
uw3_aug_rlbwxh_all="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/uw3_aug_rlbwxh_result.csv"
uw3_aug_rlbwxyh_train="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_train.csv"
uw3_aug_rlbwxyh_test="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_test.csv"
uw3_rlbwxyh_classification_result_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_classification_result.csv"

knn=KNNClassifier()
knn.classify_by_knn_tensorflow(batch=100,
                               csv_train=uw3_ori_hist_train_csv,#uw3_aug_rlbwxyh_train,
                               csv_test=uw3_ori_hist_test_csv,#uw3_aug_rlbwxyh_test,
                               k=1,
                               result_csv=uw3_ori_result_csv)#uw3_rlbwxyh_classification_result_csv)

