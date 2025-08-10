#!/usr/bin/env python3
"""
Verificador Simples de Memória CUDA
Verifica rapidamente se há memória CUDA suficiente para as transições.
"""

import torch
import sys

def verificar_memoria_cuda():
    """Verifica memória CUDA disponível"""
    if not torch.cuda.is_available():
        print("❌ CUDA não disponível")
        return False

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    memoria_total = props.total_memory
    memoria_alocada = torch.cuda.memory_allocated(device)
    memoria_livre = memoria_total - memoria_alocada
    memoria_livre_gb = memoria_livre / (1024**3)

    print(f"🎯 GPU: {props.name}")
    print(f"💾 Memória livre: {memoria_livre_gb:.2f} GB")

    if memoria_livre_gb > 2.0:
        print("✅ Memória excelente para transições")
        return True
    elif memoria_livre_gb > 1.0:
        print("🟡 Memória boa - use configurações normais")
        return True
    elif memoria_livre_gb > 0.5:
        print("🟠 Memória limitada - use configurações econômicas")
        return True
    else:
        print("🔴 Memória crítica - risco de out of memory")
        print("💡 Sugestão: Reinicie o sistema ou feche outros programas")
        return False

def limpar_memoria():
    """Limpa memória CUDA"""
    if torch.cuda.is_available():
        print("🧹 Limpando memória CUDA...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ Limpeza concluída")

if __name__ == "__main__":
    print("🔍 VERIFICAÇÃO RÁPIDA DE MEMÓRIA CUDA\n")

    if "--clean" in sys.argv:
        limpar_memoria()

    memoria_ok = verificar_memoria_cuda()

    if not memoria_ok:
        print("\n⚠️  RECOMENDAÇÕES:")
        print("1. Reinicie o sistema")
        print("2. Feche outros programas")
        print("3. Execute: python check_gpu.py --clean")

    sys.exit(0 if memoria_ok else 1)
