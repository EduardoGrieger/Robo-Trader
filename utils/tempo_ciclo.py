
import os
import csv
from time import perf_counter
from datetime import datetime

class TempoCiclo:
    """
    Instrumenta tempos do ciclo:
      - tempo_espera: tempo aguardando início de novo candle
      - tempo_total: duração total do ciclo
      - tempo_proc: tempo_total - tempo_espera
    Uso típico:
        tc = TempoCiclo()
        tc.iniciar_ciclo()
        tc.iniciar_espera()
        aguardar_inicio_novo_candle(...)
        tc.finalizar_espera()
        ... processamento ...
        tc.finalizar_ciclo()
        tc.logar(log_func, ciclo=ciclo_num)
        if config.get("telemetria_tempo_ciclo_csv", False):
            tc.salvar_csv(ciclo=ciclo_num)
    """
    def __init__(self, csv_path="logs/tempo_ciclo.csv"):
        self.csv_path = csv_path
        self.t0 = None
        self.t_total = 0.0
        self.t_espera = 0.0
        self._espera_t0 = None

    def iniciar_ciclo(self):
        self.t0 = perf_counter()
        self.t_total = 0.0
        self.t_espera = 0.0
        self._espera_t0 = None

    def iniciar_espera(self):
        # Pode ser chamado múltiplas vezes no mesmo ciclo
        self._espera_t0 = perf_counter()

    def finalizar_espera(self):
        if self._espera_t0 is not None:
            self.t_espera += (perf_counter() - self._espera_t0)
            self._espera_t0 = None

    def finalizar_ciclo(self):
        if self.t0 is None:
            return
        self.t_total = perf_counter() - self.t0

    @property
    def tempo_total(self):
        return float(self.t_total)

    @property
    def tempo_espera(self):
        return float(self.t_espera)

    @property
    def tempo_proc(self):
        # garante não-negativo por arredondamento
        return max(0.0, float(self.t_total) - float(self.t_espera))

    def logar(self, log_event_func, ciclo=None):
        try:
            msg = (f"[CICLO] tempo_total={self.tempo_total:.3f}s "
                   f"tempo_espera={self.tempo_espera:.3f}s "
                   f"tempo_proc={self.tempo_proc:.3f}s")
            if ciclo is not None:
                msg += f" (ciclo {ciclo})"
            log_event_func(msg, level="info")
        except Exception:
            pass

    def salvar_csv(self, ciclo=None):
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            escrever_cabecalho = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")
                if escrever_cabecalho:
                    w.writerow(["timestamp", "ciclo", "tempo_total_s", "tempo_espera_s", "tempo_proc_s"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), ciclo, 
                            f"{self.tempo_total:.6f}", f"{self.tempo_espera:.6f}", f"{self.tempo_proc:.6f}"])
        except Exception:
            pass
