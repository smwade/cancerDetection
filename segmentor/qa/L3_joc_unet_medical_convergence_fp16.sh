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
    --batch_size 2 \
    --benchmark \
    --exec_mode train_and_predict \
    --augment \
    --use_amp \
    --warmup_steps 200 \
    --log_every 100 \
    --max_steps 120000 > /results/qa_log.txt

if [[ $? -ne 0 ]]; then
    cat /results/qa_log.txt
    echo TRAINING SCRIPT FAILED
    exit 1
fi

LOSS=$(grep 'dice_loss' /results/qa_log.txt | tail -1 | cut -d '"' -f2 )

if [[ -z "LOSS" ]]; then
    cat /results/qa_log.txt
    echo UNEXPECTED END OF LOG
    exit 1
fi

BASELINE=0.1

echo REACHED $LOSS loss

if [[ $LOSS > $BASELINE ]]; then
    cat /results/qa_log.txt
    echo FAILED TO REACH EXPECTED ACCURACY FOR BS=2
    exit 1
fi

cat /results/qa_log.txt
echo SUCCESS
exit 0
