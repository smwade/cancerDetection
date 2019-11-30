echo -------------------------------------------------------------------
echo SMOKE TEST FOR UNET MEDICAL - TRAIN AND PREDICT FP32 - BS=1 - 8 GPU
echo -------------------------------------------------------------------

mpirun \
    -np 8 \
    -H localhost:8 \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -mca pml ob1 -mca btl ^openib \
    --allow-run-as-root \
    python main.py \
    --data_dir /data/unet_medical_tf \
    --model_dir /results \
    --batch_size 1 \
    --exec_mode train_and_predict \
    --augment \
    --max_steps 1 > /results/qa_log.txt

if [[ $? -ne 0 ]]; then
    cat /results/qa_log.txt
    echo LOG SCRIPT NOT FOUND
    exit 1
fi

echo SUCCESS
exit 0