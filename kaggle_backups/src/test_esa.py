
from pathlib import Path
from utils_users import TomatoProblem

# Gjejme skedarin easy.kttsp
rruga = Path(__file__).resolve().parent.parent / "data" / "easy.kttsp"

# Krijojme objektin e problemit
problemi = TomatoProblem(str(rruga))

# Shfaqim informacionet
print("Numri i domateve:", problemi.n_tomatoes)
print("Koha maksimale:", problemi.max_time)
print("Kufiri normal DV:", problemi.dv_threshold)
print("Kufiri special DV:", problemi.exception_threshold)
print("Manovrat speciale:", problemi.n_exceptions)
