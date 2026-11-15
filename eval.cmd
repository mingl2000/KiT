https://github.com/Luciferbobo/KiT

uv run  eval/infer_for_pred.py --case 300880.SZ,5m,2026-03-04  --ckpt .\ckpt\step_market_284M\step_market_284M.pt
uv run python eval/cal_metrics.py