echo --------------------------------------------
echo SMOKE TEST FOR UNET MEDICAL - PREDICT - BS=1
echo --------------------------------------------

python main.py \
--data_dir /data/unet_medical_tf \
--model_dir /results \
--batch_size 1 \
--exec_mode predict \
--augment \
--warmup_steps 0 \
--log_every 1 \
--max_steps 1 > /results/qa_log.txt

if [[ $? -ne 0 ]]; then
    cat /results/qa_log.txt
    echo LOG SCRIPT NOT FOUND
    exit 1
fi

echo SUCCESS
exit 0