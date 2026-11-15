@echo off
rem Rebuild data_yahoo/ (same layout as data/) from Yahoo Finance for every ticker in data/instruments.json.
rem Extra arguments are passed through, e.g.:  download_yahoo.cmd --tickers 600031.SH --scales 1d,30m
uv run --extra yahoo python scripts/download_yahoo_ashare.py %*
