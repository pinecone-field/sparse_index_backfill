pip uninstall urllib3
pip install pinecone urllib3
pip install -U "pinecone[grpc]"
python sparseBackfill_async_grpc.py \
  --index sparse-backfill-test1 \
  --namespace ccnews_parquet_fixed-250k \
  --text-metadata-field text \
  --page-workers 4 \
  --embed-workers 4 \
  --update-workers 16 \
  --embed-batch-size 96
