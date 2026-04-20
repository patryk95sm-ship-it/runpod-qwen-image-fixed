# Qwen Image RunPod Handler (Fixed)

Działający handler dla Qwen Image na RunPod Serverless. **Nie zawiera błędu `cross_attention_kwargs`.**

## Szybkie wdrożenie

### 1. Stwórz repo na GitHub
Wgraj te pliki do nowego repozytorium GitHub:
- `handler.py`
- `Dockerfile`
- `requirements.txt`
- `README.md`

### 2. Podlinkuj do RunPod
1. Wejdź w [RunPod Console](https://www.runpod.io/console/serverless)
2. Kliknij **New Endpoint**
3. Wybierz **Deploy from GitHub**
4. Wybierz swoje nowe repo
5. Ustaw:
   - **Branch:** `main`
   - **Dockerfile Path:** `Dockerfile`
   - **Container Disk:** `20 GB`
   - **Network Volume:** Utwórz nowy (~100 GB) i podłącz go — model Qwen to ~57GB
   - **GPU:** `RTX 4090`, `A100`, `H100` (zalecane 48GB+ VRAM)
   - **Workers:** 0-3
   - **Timeout:** 600

6. Kliknij **Deploy**

### 3. Poczekaj na build
Pierwszy build może trwać 10-15 minut. Worker pobierze model z HuggingFace na Network Volume.

## API Input

```json
{
  "input": {
    "prompt": "a serene mountain landscape at sunset",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": 42
  }
}
```

## API Output

```json
{
  "image": "base64_encoded_png...",
  "seed": 42
}
```

## Co było naprawione?

Usunięto przekazywanie `cross_attention_kwargs` do `QwenImagePipeline.__call__()`, które powodowało błąd:
```
TypeError: QwenImagePipeline.__call__() got an unexpected keyword argument 'cross_attention_kwargs'
```

Qwen Image użyje własnych mechanizmów attention bez potrzeby manualnego podawania `cross_attention_kwargs`.
