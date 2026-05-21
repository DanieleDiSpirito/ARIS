import time
import os
import csv
import psutil
import asyncio
from functools import wraps

def misura_performance(metodo, output_csv=None):
    """
    Decoratore per misurare Tempo, RAM e VRAM di una funzione di parsing.
    I risultati vengono accodati in un CSV pronto per l'analisi della tesi.
    Supporta sia funzioni sincrone che asincrone.
    """
    if output_csv is None:
        # Calcola il percorso assoluto rispetto a telemetry.py per robustezza rispetto a CWD
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_csv = os.path.abspath(os.path.join(current_dir, "../../data/metrics/benchmark_loaders.csv"))

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(filepath, *args, **kwargs):
                # 1. Setup VRAM GPU (Se stiamo usando l'accelerazione hardware via CUDA)
                gpu_available = False
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_available = True
                        torch.cuda.reset_peak_memory_stats()
                except ImportError:
                    pass

                # 2. Fotografia iniziale della RAM di sistema
                process = psutil.Process(os.getpid())
                ram_start = process.memory_info().rss / (1024 * 1024) # Convertito in MB

                # 3. Avvio Cronometro
                start_time = time.perf_counter()

                # ==========================================
                # ESECUZIONE DELLA FUNZIONE DI ESTRAZIONE VERA E PROPRIA (ASYNC)
                risultato = await func(filepath, *args, **kwargs)
                # ==========================================

                # 4. Fine Cronometro
                end_time = time.perf_counter()
                tempo_impiegato = end_time - start_time

                # 5. Calcolo consumi (Delta)
                ram_end = process.memory_info().rss / (1024 * 1024)
                ram_usata = max(0, ram_end - ram_start)

                vram_usata = 0
                if gpu_available:
                    vram_usata = torch.cuda.max_memory_allocated() / (1024 * 1024)

                # 6. Salvataggio su file CSV per i grafici della Tesi
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                file_exists = os.path.isfile(output_csv)

                with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["File", "Metodo", "Tempo (s)", "RAM (MB)", "VRAM (MB)"])
                    
                    nome_file = os.path.basename(filepath)
                    writer.writerow([nome_file, metodo, round(tempo_impiegato, 2), round(ram_usata, 2), round(vram_usata, 2)])

                return risultato
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(filepath, *args, **kwargs):
                # 1. Setup VRAM GPU (Se stiamo usando l'accelerazione hardware via CUDA)
                gpu_available = False
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_available = True
                        torch.cuda.reset_peak_memory_stats()
                except ImportError:
                    pass

                # 2. Fotografia iniziale della RAM di sistema
                process = psutil.Process(os.getpid())
                ram_start = process.memory_info().rss / (1024 * 1024) # Convertito in MB

                # 3. Avvio Cronometro
                start_time = time.perf_counter()

                # ==========================================
                # ESECUZIONE DELLA FUNZIONE DI ESTRAZIONE VERA E PROPRIA (SYNC)
                risultato = func(filepath, *args, **kwargs)
                # ==========================================

                # 4. Fine Cronometro
                end_time = time.perf_counter()
                tempo_impiegato = end_time - start_time

                # 5. Calcolo consumi (Delta)
                ram_end = process.memory_info().rss / (1024 * 1024)
                ram_usata = max(0, ram_end - ram_start)

                vram_usata = 0
                if gpu_available:
                    vram_usata = torch.cuda.max_memory_allocated() / (1024 * 1024)

                # 6. Salvataggio su file CSV per i grafici della Tesi
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                file_exists = os.path.isfile(output_csv)

                with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["File", "Metodo", "Tempo (s)", "RAM (MB)", "VRAM (MB)"])
                    
                    nome_file = os.path.basename(filepath)
                    writer.writerow([nome_file, metodo, round(tempo_impiegato, 2), round(ram_usata, 2), round(vram_usata, 2)])

                return risultato
            return wrapper
    return decorator