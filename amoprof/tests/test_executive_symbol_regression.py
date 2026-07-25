from pathlib import Path

def test_ssd_bw_pct_initializer_before_first_use():
    p = Path(__file__).resolve().parents[1] / "amoprof" / "report" / "combined.py"
    txt = p.read_text()
    init = txt.find("_ssd_bw_target_mbs = 7000.0")
    use = txt.find("BW target pct")
    assert init >= 0 and use >= 0 and init < use
