from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_native_negative_and_interpretation_scope():
 text=(ROOT/'papers/parkinson.tex').read_text()
 assert 'five ON and one OFF' in text and 'not physiological validation' in text
 assert 'the one nonlinearity PID cannot track' not in text
 assert 'not a current final count' in text
 assert '\\input{native_multiseed_sec}' in text and '\\input{native_dutycycle_sec}' in text
