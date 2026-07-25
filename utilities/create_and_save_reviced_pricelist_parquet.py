from pathlib import Path

from wequant.data_processing import PricelistPl, read_data


CURRENT_DIR = Path(__file__).resolve().parent
PJ_DIR = CURRENT_DIR.parent

# dataの読み込み
fp = PJ_DIR / "data" / "raw_pricelist.parquet"
df = read_data(fp)

# dataを分割修正して保存
RawPL = PricelistPl(df)
reviced_priceliset_df = RawPL.get_reviced_pricelist()
reviced_priceliset_df.write_parquet(PJ_DIR / "data" / "reviced_pricelist.parquet")