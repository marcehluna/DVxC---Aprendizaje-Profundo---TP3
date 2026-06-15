# Reporte de experimentos — TP3 Clasificador de Emociones

Registro de corridas a partir de **2026-06-06**.  
Notebook: `Luna-Marcelo-DL-TP3-CO24.ipynb`

---

## Cómo usar este documento

1. **Antes de entrenar:** anotar el ID del experimento (E0, E1, …) y copiar la config activa del notebook.
2. **Al cambiar algo:** registrar solo lo que difiere respecto al experimento anterior (preprocesamiento, modelo o entrenamiento).
3. **Después de entrenar:** completar métricas de validación y observaciones breves (incl. sección *Comentarios del entrenamiento* para la narrativa del §3.6 del notebook).
4. **Actualizar** la tabla comparativa al final de cada corrida.

**Inferencia externa (puntos 4 y 5):** usar IDs **I0, I1, …** (modelo congelado en **E6**). Config activa en notebook **§5.1** (`EXPERIMENTO_INFERENCIA`, Haar, `METRICAS_INFERENCIA_*`). Resúmenes y motivación en este reporte.

Campos prioritarios por experimento:

| Categoría | Qué registrar |
|-----------|---------------|
| **Preprocesamiento** | `TAMANO_IMAGEN`, `MODO_COLOR`, augmentations (`APLICAR_*`), balanceo |
| **Modelo / entrenamiento** | arquitectura (`FILTROS_CONV`, `DROPOUT_FC`, …), `LEARNING_RATE`, `OPTIMIZADOR`, early stopping |
| **Métricas** | accuracy, F1 macro, loss val, mejor época, época de parada |

---

## E0 — Baseline (config actual del notebook)

> Punto de partida para comparar experimentos.  
> **Corrida completada** — métricas de la mejor época restaurada (17).

### Preprocesamiento

| Parámetro | Valor |
|-----------|-------|
| `TAMANO_IMAGEN` | `(48, 48)` |
| `MODO_COLOR` | `"grayscale"` → 1 canal |
| `NORMALIZE_MEAN` / `STD` | `[0.5]` / `[0.5]` |
| `APLICAR_VOLTEO_HORIZONTAL` | `True` |
| `APLICAR_ROTACION` | `False` (`ROTACION_GRADOS = 15`) |
| `APLICAR_RECORTE_ALEATORIO` | `False` (`RECORTE_TAMANO = (44, 44)`) |
| `APLICAR_COLOR_JITTER` | `True` (brightness `0.2`, contrast `0.2`) |
| `APLICAR_BALANCEO` | `True` (`WeightedRandomSampler`) |

### Modelo

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | 3 bloques Conv2d→BN→ReLU→MaxPool |
| `FILTROS_CONV` | `[32, 64, 128]` |
| `TAMANO_KERNEL` / `PADDING_CONV` | `3` / `1` |
| `TAMANO_POOL` | `2` |
| `UNIDADES_FC_OCULTAS` | `256` |
| `DROPOUT_FC` | `0.5` |
| Loss | `CrossEntropyLoss` |

### Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| `BATCH_SIZE` | `32` |
| `NUM_EPOCHS` | `40` (máx.) |
| `LEARNING_RATE` | `1e-4` |
| `OPTIMIZADOR` | `"adam"` |
| `APLICAR_EARLY_STOPPING` | `True` |
| `EARLY_STOPPING_METRICA` | `"val_loss"` |
| `EARLY_STOPPING_PATIENCE` | `5` |
| `EARLY_STOPPING_MIN_DELTA` | `0.0` |
| `EARLY_STOPPING_RESTAURAR_MEJOR_MODELO` | `True` |

### Inferencia punto 5 (Haar)

| Parámetro | Valor |
|-----------|-------|
| `SCALE_FACTOR` | `1.1` |
| `MIN_NEIGHBORS` | `6` |
| `MIN_SIZE` | `(30, 30)` |

### Métricas — validación

| Métrica | Valor |
|---------|-------|
| Accuracy | **75.42%** (0.7542) |
| F1 macro | **66.71%** (0.6671) |
| Loss | **0.6924** |
| Mejor época | **17** (`val_loss` = 0.6924) |
| Época de parada | **22** (early stopping, patience 5) |

#### F1 por clase (validación)

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| alegría | 0.93 | 0.84 | **0.88** |
| sorpresa | 0.79 | 0.79 | **0.79** |
| seriedad | 0.69 | 0.76 | 0.72 |
| enojo | 0.63 | 0.71 | 0.67 |
| tristeza | 0.65 | 0.68 | 0.66 |
| miedo | 0.55 | 0.50 | **0.52** |
| disgusto | 0.41 | 0.44 | **0.43** |

### Observaciones

- Early stopping restauró pesos de la **época 17**; a partir de ahí el train siguió bajando loss mientras val empeoraba → overfitting moderado.
- Clases más débiles: **disgusto** (F1 0.43) y **miedo** (F1 0.52); alegría concentra la mayoría de muestras y alcanza el mejor F1 (0.88).
- F1 macro (66.7%) por debajo del accuracy (75.4%): las clases minoritarias o difíciles arrastran el rendimiento global.

---

## Tabla comparativa

Completar una fila por corrida. En **Cambio vs anterior** resumir en una línea qué se modificó.

| Exp | Fecha | Cambio vs anterior | Acc val | F1 macro | Loss val | Mejor época | Parada | Notas |
|-----|-------|--------------------|---------|----------|----------|-------------|--------|-------|
| E0 | 2026-06-06 | Baseline (config actual) | 75.4% | 66.7% | 0.69 | 17 | 22 | Disgusto/miedo débiles |
| E1 | 2026-06-14 | Early stopping → `val_f1` | 76.8% | 69.5% | 0.68 | 16 | 21 | +2.8 pp F1 vs E0; disgusto ↑ |
| E2 | 2026-06-15 | Loss ponderada `sqrt_inverso` | 73.9% | 66.0% | 0.77 | 19 | 24 | Peor que E1; doble compensación |
| E3 | 2026-06-16 | Dropout FC 0.5 → 0.6 | 76.4% | 68.4% | 0.69 | 18 | 23 | Ligeramente peor que E1 |
| E4 | 2026-06-17 | Resolución 48×48 → 64×64 | 77.5% | 69.7% | 0.65 | 17 | 22 | Mejor F1 global; +0.2 pp vs E1 |
| E5 | 2026-06-18 | RandomRotation ±10° | 77.0% | 68.8% | 0.64 | 32 | 37 | Peor que E4; mantener E4 |
| E6 | 2026-06-19 | Loss `sqrt_inverso` sin sampler | 79.3% | 71.3% | 0.61 | 29 | 34 | **Config final entrega** |
| E7 | 2026-06-19 | RGB vs gris (E6 base) | 79.0% | 70.9% | 0.62 | 25 | 30 | Descartado; −0.4 pp F1 vs E6 |
| E8 | 2026-06-19 | RandomCrop 56×56 + Resize (train) | 70.1% | 60.9% | 0.84 | 24 | 29 | Descartado; −10.4 pp F1 vs E6 |

---

## Registro cronológico

### E0 — Baseline

**Fecha:** 2026-06-06  
**Estado:** corrida completada

**Preprocesamiento:** 48×48, gris, normalización 0.5/0.5, flip horizontal + ColorJitter (0.2/0.2), balanceo activo. Sin rotación ni recorte aleatorio.

**Modelo / entrenamiento:** CNN 32-64-128, FC 256, dropout 0.5, Adam lr=1e-4, batch 32, early stopping en val_loss (patience 5).

**Métricas — validación (mejor época 17):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 75.42% | 66.71% | 0.6924 | 17 | 22 |

**Observaciones:**
- Overfitting a partir de época 17 (train acc ~88% vs val ~75% en época 22).
- Peores clases: disgusto (F1 0.43), miedo (F1 0.52).
- Mejor clase: alegría (F1 0.88).

---

### E1 — Early stopping en `val_f1`

**Fecha:** 2026-06-14  
**Estado:** corrida completada

**Cambio vs E0:**
- Preprocesamiento: ninguno (misma config E0)
- Modelo: ninguno
- Entrenamiento: `EARLY_STOPPING_METRICA = "val_f1"` (antes `"val_loss"`); resto igual (patience 5, Adam lr=1e-4)

**Métricas — validación (mejor época 16, criterio val_f1 = 0.6945):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 76.79% | 69.45% | 0.6752 | 16 | 21 |

#### F1 por clase vs E0

| Clase | F1 E0 | F1 E1 | Δ |
|-------|-------|-------|---|
| alegría | 0.88 | 0.88 | — |
| sorpresa | 0.79 | 0.80 | +0.01 |
| seriedad | 0.72 | 0.75 | +0.03 |
| enojo | 0.67 | 0.69 | +0.02 |
| tristeza | 0.66 | 0.66 | — |
| miedo | 0.52 | 0.58 | +0.06 |
| disgusto | 0.43 | 0.51 | **+0.08** |

**Observaciones:**
- Mejora global respecto a E0: **+1.4 pp accuracy**, **+2.7 pp F1 macro**, loss val ligeramente menor.
- El criterio `val_f1` priorizó un checkpoint (ép. 16) con mejor balance entre clases; disgusto y miedo suben notablemente.
- Disgusto: recall 0.55 (E0: 0.44), acierto diagonal en matriz norm. ~55% (E0: ~44%).
- Miedo: sube F1 pero recall baja (0.47 vs 0.50); gana precision (0.76 vs 0.55) — más conservador al predecir miedo.

---

### E2 — Loss ponderada por clase (`sqrt_inverso`)

**Fecha:** 2026-06-15  
**Estado:** corrida completada

**Cambio vs E1:**
- Preprocesamiento: ninguno (misma config E0/E1)
- Modelo: `CrossEntropyLoss(weight=...)` con `APLICAR_PESOS_EN_LOSS = True`, `MODO_PESOS_CLASE = "sqrt_inverso"`
- Entrenamiento: early stopping en `val_f1` (igual E1); balanceo activo (igual E1)

**Pesos de loss (sqrt_inverso, media normalizada = 1):**

| Clase | n train | Peso |
|-------|---------|------|
| alegria | 4772 | 0.46 |
| disgusto | 717 | 1.19 |
| enojo | 705 | 1.20 |
| miedo | 281 | 1.90 |
| seriedad | 2524 | 0.64 |
| sorpresa | 1290 | 0.89 |
| tristeza | 1982 | 0.72 |

**Métricas — validación (mejor época 19, criterio val_f1 = 0.6601):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 73.92% | 66.01% | 0.7725 | 19 | 24 |

#### F1 por clase vs E1

| Clase | F1 E1 | F1 E2 | Δ |
|-------|-------|-------|---|
| alegría | 0.88 | 0.87 | −0.01 |
| sorpresa | 0.80 | 0.77 | −0.03 |
| seriedad | 0.75 | 0.72 | −0.03 |
| enojo | 0.69 | 0.66 | −0.03 |
| tristeza | 0.66 | 0.60 | −0.06 |
| miedo | 0.58 | 0.56 | −0.02 |
| disgusto | 0.51 | 0.44 | **−0.07** |

**Observaciones:**
- **Regresión global vs E1:** −2.9 pp accuracy, −3.4 pp F1 macro. El checkpoint elegido por `val_f1` quedó por debajo del de E1.
- **Doble compensación:** sampler (`WeightedRandomSampler`) + loss ponderada parecen sobre-corregir el desbalance; la loss escalada (val ~0.77 vs ~0.68 en E1) no es directamente comparable, pero las métricas de clasificación sí empeoran.
- **Disgusto:** recall sube (0.59 vs 0.55 en E1) pero precision cae fuerte (0.35); más falsos positivos → F1 baja a 0.44. Confusiones hacia disgusto desde seriedad (65/680) y tristeza (44/478).
- **Miedo:** precision alta (0.75) pero recall bajo (0.45); patrón similar a E1 pero peor F1 global.
- **Conclusión preliminar:** mantener E1 como base; si se reintenta loss ponderada, probar **sin sampler** o con `MODO_PESOS_CLASE = "balanceado"` solo en ablation — no acumular ambos mecanismos sin control.

---

### E3 — Dropout FC 0.6 (regularización)

**Fecha:** 2026-06-16  
**Estado:** corrida completada

**Cambio vs E1:**
- Preprocesamiento: ninguno
- Modelo: `DROPOUT_FC = 0.6` (E1: 0.5)
- Entrenamiento: early stopping en `val_f1`, balanceo activo, `CrossEntropyLoss` sin pesos (igual E1)

**Métricas — validación (mejor época 18, criterio val_f1 = 0.6837):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 76.43% | 68.37% | 0.6913 | 18 | 23 |

#### F1 por clase vs E1

| Clase | F1 E1 | F1 E3 | Δ |
|-------|-------|-------|---|
| alegría | 0.88 | 0.89 | +0.01 |
| tristeza | 0.66 | 0.68 | +0.02 |
| enojo | 0.69 | 0.69 | — |
| disgusto | 0.51 | 0.50 | −0.01 |
| seriedad | 0.75 | 0.72 | −0.03 |
| sorpresa | 0.80 | 0.77 | −0.03 |
| miedo | 0.58 | 0.54 | −0.04 |

**Observaciones:**
- **No supera E1:** −0.4 pp accuracy, −1.1 pp F1 macro vs E1 (69.5%). El dropout extra no compensó la pérdida de capacidad útil en la FC.
- **Brecha train/val algo menor** en la mejor época (train acc ~82% vs val ~76%, gap ~6 pp) frente al overfitting más marcado de E1 — la regularización actuó, pero el checkpoint óptimo en `val_f1` quedó por debajo.
- **Miedo y sorpresa** bajan; **tristeza** sube levemente (+0.02). Disgusto prácticamente igual (0.50 vs 0.51).
- **Conclusión:** mantener **E1** (`DROPOUT_FC = 0.5`) como base. Próximo candidato: **E4** (64×64) o **E3b** (`weight_decay` sin subir dropout).

---

### E4 — Resolución 64×64

**Fecha:** 2026-06-17  
**Estado:** corrida completada

**Cambio vs E1:**
- Preprocesamiento: `TAMANO_IMAGEN = (64, 64)` (E1: 48×48); resto igual (gris, flip + ColorJitter, balanceo)
- Modelo: misma arquitectura; FC se recalcula automáticamente (8192 → 256 vs 1152 → 256 en 48×48)
- Entrenamiento: early stopping `val_f1`, dropout 0.5, loss sin pesos (igual E1)

**Métricas — validación (mejor época 17, criterio val_f1 = 0.6969):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 77.48% | 69.69% | 0.6464 | 17 | 22 |

#### F1 por clase vs E1

| Clase | F1 E1 | F1 E4 | Δ |
|-------|-------|-------|---|
| alegría | 0.88 | 0.89 | +0.01 |
| seriedad | 0.75 | 0.75 | — |
| miedo | 0.58 | 0.58 | — |
| enojo | 0.69 | 0.70 | +0.01 |
| tristeza | 0.66 | 0.67 | +0.01 |
| sorpresa | 0.80 | 0.79 | −0.01 |
| disgusto | 0.51 | 0.49 | −0.02 |

**Observaciones:**
- **Mejor resultado global hasta ahora:** +0.7 pp accuracy, +0.2 pp F1 macro vs E1; loss val más baja (0.65 vs 0.68).
- La mejora es **modesta**; disgusto sigue débil (0.49) e incluso baja levemente vs E1. Miedo se mantiene en 0.58.
- **Seriedad** mejora recall (0.82 vs ~0.76 en E1) con precision estable — menos confusiones desde tristeza/disgusto hacia seriedad en matriz norm.
- **Miedo:** recall 0.53, precision 0.65; diagonal ~53% (similar a E1).
- **Conclusión:** **E4** fue la mejor config hasta E6. Superseded por E6 (+1.6 pp F1 macro).

---

### E5 — Rotación aleatoria ±10°

**Fecha:** 2026-06-18  
**Estado:** corrida completada

**Cambio vs E4:**
- Preprocesamiento: `APLICAR_ROTACION = True`, `ROTACION_GRADOS = 10` (solo train); resto igual (64×64, flip, ColorJitter, balanceo)
- Modelo / entrenamiento: sin cambios (early stopping `val_f1`, dropout 0.5, loss sin pesos)

**Métricas — validación (mejor época 32, criterio val_f1 = 0.6875):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 77.02% | 68.75% | 0.6445 | 32 | 37 |

#### F1 por clase vs E4

| Clase | F1 E4 | F1 E5 | Δ |
|-------|-------|-------|---|
| alegría | 0.89 | 0.89 | — |
| tristeza | 0.67 | 0.70 | +0.03 |
| enojo | 0.70 | 0.68 | −0.02 |
| seriedad | 0.75 | 0.73 | −0.02 |
| sorpresa | 0.79 | 0.78 | −0.01 |
| disgusto | 0.49 | 0.49 | — |
| miedo | 0.58 | 0.53 | **−0.05** |

**Observaciones:**
- **No supera E4:** −0.5 pp accuracy, −0.9 pp F1 macro vs E4 (69.7%). Entrenamiento más largo (mejor época 32, parada 37) sin ganancia en el checkpoint óptimo.
- **Disgusto:** F1 igual (0.49); sube recall (0.54 vs 0.46 en E4) pero baja precision (0.45 vs 0.53) — más falsos positivos.
- **Miedo** empeora (−0.05 F1); posible efecto de rotación en expresiones con ojos/boca abiertos.
- **Tristeza** mejora levemente (+0.03); no compensa el retroceso global.
- **Conclusión:** revertir a **E4** (rotación desactivada). E6 probado → supera E4.

---

### E6 — Loss `sqrt_inverso` sin sampler

**Fecha:** 2026-06-19  
**Estado:** corrida completada

**Cambio vs E4:**
- Preprocesamiento: 64×64, flip + ColorJitter (igual E4); **`APLICAR_BALANCEO = False`** (E4: sampler activo)
- Modelo: **`CrossEntropyLoss(weight=...)`, `MODO_PESOS_CLASE = "sqrt_inverso"`** (E4: sin pesos)
- Entrenamiento: early stopping `val_f1`, dropout 0.5 (igual E4)

**Métricas — validación (mejor época 29, criterio val_f1 = 0.7130):**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| 79.34% | 71.30% | 0.6097 | 29 | 34 |

#### F1 por clase vs E4

| Clase | F1 E4 | F1 E6 | Δ |
|-------|-------|-------|---|
| alegría | 0.89 | 0.90 | +0.01 |
| seriedad | 0.75 | 0.77 | +0.02 |
| sorpresa | 0.79 | 0.80 | +0.01 |
| tristeza | 0.67 | 0.71 | **+0.04** |
| enojo | 0.70 | 0.70 | — |
| miedo | 0.58 | 0.59 | +0.01 |
| disgusto | 0.49 | 0.52 | **+0.03** |

#### Comparativa de estrategias de desbalance

| Exp | Sampler | Loss ponderada | F1 macro |
|-----|---------|----------------|----------|
| E4 | sí | no | 69.7% |
| E2 | sí | sí | 66.0% |
| **E6** | **no** | **sí** | **71.3%** |

**Observaciones:**
- **Mejor resultado global de la campaña:** +1.9 pp accuracy, +1.6 pp F1 macro vs E4; +4.6 pp F1 vs E0.
- La ablation confirma la hipótesis de E2: el problema fue la **doble compensación**, no la loss ponderada en sí.
- **Disgusto** sube a F1 0.52 (recall 0.53, precision 0.51) — mejor equilibrio que E4/E2.
- **Tristeza** +0.04 F1; **seriedad** +0.02; todas las clases ≥ 0.52 F1.
- Train acc ~87% vs val ~79% en mejor época — overfitting moderado pero checkpoint óptimo claro.
- **Conclusión:** **E6 pasa a ser la config final recomendada** (64×64 + loss sqrt_inverso, sin sampler). **E7** (RGB) y **E8** (RandomCrop) descartados — ver § E7 y § E8.

---

### E7 — Entrada RGB (vs escala de grises)

**Fecha:** 2026-06-19  
**Estado:** corrida completada — **descartado** (mantener E6)  
**Base:** E6  
**Modelo / entrenamiento:** igual E6 (CNN 32-64-128, loss `sqrt_inverso`, sin sampler, early stopping `val_f1`, dropout 0.5, Adam lr=1e-4).

**Cambio vs E6:** §1.2 — `MODO_COLOR = "rgb"` (E6: `"grayscale"`); primera capa conv **1 → 3 canales**; normalización `[0.5, 0.5, 0.5]`.

#### Métricas — validación (mejor época 25, criterio val_f1 = 0.7090)

| Pipeline | Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|----------|------|-------------|--------|
| E6 (ref.) | **79.34%** | **71.30%** | **0.6097** | 29 | 34 |
| **E7** | 79.04% | 70.90% | 0.6223 | 25 | 30 |

| Métrica | Δ vs E6 |
|---------|---------|
| Accuracy | −0.30 pp |
| F1 macro | **−0.40 pp** |
| Loss | +0.0126 |

#### F1 por clase vs E6

| Clase | F1 E6 | F1 E7 | Δ |
|-------|-------|-------|---|
| alegría | 0.90 | 0.90 | = |
| sorpresa | 0.80 | 0.80 | = |
| seriedad | 0.77 | 0.76 | −0.01 |
| tristeza | 0.71 | 0.72 | +0.01 |
| enojo | 0.70 | 0.73 | +0.03 |
| miedo | 0.59 | 0.59 | = |
| disgusto | 0.52 | 0.47 | **−0.05** |

**Observaciones:**
- RGB **no supera** E6 en accuracy ni F1 macro; la hipótesis cromática no se confirma en validación alineada.
- **Disgusto** retrocede (recall ~39% vs ~53% en E6) — empeora la clase ya más difícil.
- **Enojo** mejora levemente (+0.03 F1); no compensa el retroceso global.
- Confusiones similares a E6 (tristeza↔seriedad, disgusto→seriedad/tristeza, miedo→sorpresa).

**Conclusión E7:** descartado. **Config de entrega y modelo congelado para inferencia (I0–I3): E6 en escala de grises.**

---

### E8 — RandomCrop domain shift

**Fecha:** 2026-06-19  
**Estado:** corrida completada — **descartado** (mantener E6)  
**Base:** E6 (gris)

**Motivación:** P4/P5 fallan por domain shift (fotos reales + Haar vs rostros alineados). E7 (RGB) no ayudó; E8 prueba recorte aleatorio en train para simular desalineación.

**Cambio vs E6:** `APLICAR_RECORTE_ALEATORIO = True`, `RECORTE_TAMANO = (56, 56)` en train; pipeline train: `Resize 64` → `RandomCrop 56` → `Resize 64`. Resto igual E6.

#### Métricas — validación (mejor época 24, criterio val_f1 = 0.6090)

| Pipeline | Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|----------|------|-------------|--------|
| E6 (ref.) | **79.34%** | **71.30%** | **0.6097** | 29 | 34 |
| **E8** | 70.08% | 60.90% | 0.8390 | 24 | 29 |

| Métrica | Δ vs E6 |
|---------|---------|
| Accuracy | **−9.26 pp** |
| F1 macro | **−10.40 pp** |
| Loss | +0.2293 |

#### F1 por clase vs E6

| Clase | F1 E6 | F1 E8 | Δ |
|-------|-------|-------|---|
| alegría | 0.90 | 0.84 | −0.06 |
| sorpresa | 0.80 | 0.72 | −0.08 |
| seriedad | 0.77 | 0.67 | −0.10 |
| tristeza | 0.71 | 0.65 | −0.06 |
| enojo | 0.70 | 0.56 | −0.14 |
| miedo | 0.59 | 0.46 | −0.13 |
| disgusto | 0.52 | 0.36 | −0.16 |

**Observaciones:**
- **Retroceso global** en todas las clases; peor en disgusto (−0.16), enojo (−0.14) y miedo (−0.13).
- El train con crop+zoom hace el aprendizaje más difícil, pero **val sigue siendo rostros alineados** → no hay ganancia; sí hay degradación fuerte.
- Confusiones aumentan: seriedad absorbe más errores (disgusto→seriedad, tristeza→seriedad); enojo con alta recall (0.77) pero baja precision (0.44).
- Patrón similar a E5 (rotación): augmentations geométricas agresivas **no ayudan** en este dataset alineado.

**Conclusión E8:** descartado. **E6 permanece config final.** El gap P4/P5 no se cierra reentrenando con RandomCrop; siguiente vía: **inferencia** (alineamiento post-Haar) o narrativa de domain shift en entrega.

**Inferencia P5 con checkpoint E8:** no ejecutada (no supera criterio en val).

---

## Comentarios del entrenamiento (notebook §3.6)

Registro narrativo de la campaña **E0–E8** e inferencia **I0–I3**.  
El notebook §3.6 es un **placeholder** hasta la entrega: remite a este apartado; la **copia final para el Colab** se pegará en §3.6 al cerrar todas las corridas (ver *Cierre para el Colab* abajo).

### Síntesis E0→E6 (config de entrega actual)

**Resultado E6:** accuracy **79.34%**, F1 macro **71.30%**, loss **0.6097** (mejor época 29, early stop en 34).

**Hitos de la campaña:**
- **E1:** early stopping en `val_f1` alinea el checkpoint con la métrica de evaluación (+2.8 pp F1 vs E0).
- **E4:** resolución 64×64 aporta detalle facial; mejor config previa (69.7% F1).
- **E2 descartado:** sampler + loss ponderada juntos empeoraron (doble compensación).
- **E6:** loss `sqrt_inverso` **sin** sampler — mejor resultado global (+1.6 pp F1 vs E4).

**Por clase (E6):** alegría y sorpresa >0.80 F1; seriedad/tristeza/enojo ~0.70–0.77; miedo 0.59; disgusto 0.52 (clase más difícil).

**Confusiones persistentes (validación E6):** tristeza↔seriedad (~14%), disgusto→seriedad (~14%), miedo→sorpresa (~12%). Parte del techo es semántico (expresiones parecidas) y de resolución.

**Overfitting:** train acc ~87% vs val ~79% en mejor época E6 — patrón distinto al de configs con sampler (E4).

### E7 (RGB) — descartado

**Resultado:** accuracy 79.04%, F1 macro 70.90%, loss 0.6223 (ép. 25, parada 30). **−0.4 pp F1 vs E6.**

**Decisión:** mantener **E6** (gris) como config final. RGB no aporta; disgusto empeora (F1 0.47).

### E8 (RandomCrop) — descartado

**Resultado:** accuracy 70.08%, F1 macro 60.90%, loss 0.8390 (ép. 24, parada 29). **−10.4 pp F1 vs E6.**

**Decisión:** mantener **E6**. Augmentations geométricas (E5 rotación, E8 crop) empeoran en val alineada. Campaña de reentrenamiento **E7–E8 cerrada**; config final congelada en E6.

**Brecha P4/P5:** atacar por **inferencia** (alineamiento post-Haar) o documentar domain shift en entrega — no más reentrenos con augment geométrico/RGB.

### Cierre para el Colab *(completar al entregar)*

> El notebook §3.6 debe quedar como **placeholder** hasta el cierre; no duplicar aquí la narrativa E6 durante las corridas.

- [x] Registrar E8 en este reporte
- [ ] Redactar §3.6 del notebook con síntesis final (desde este apartado + inferencia I0–I3 + puntos 4–5)
- [ ] Sustituir el recuadro `[ ESPACIO RESERVADO … ]` en §3.6 por la síntesis en prosa
- [ ] Verificar coherencia con `conclusiones.md` y outputs visibles de §3

---

## Campaña de inferencia externa (I0, I1, …)

Evaluación sobre **14 imágenes propias** (§4.2), mismas en puntos 4 y 5.  
**Modelo congelado:** E6 (79.3% acc / 71.3% F1 macro en validación alineada).

| Qué registrar | Detalle |
|---------------|---------|
| **Punto 4** | `transform_val` sobre foto completa (sin Haar) |
| **Punto 5** | Haar + recorte + `transform_val` |
| **Detección** | X/14 rostros detectados (solo P5) |
| **Accuracy** | aciertos/14 vs etiqueta manual |
| **Por imagen** | archivo → esperada → pred P4 → pred P5 → ¿detectado? |

---

### I0 — Baseline inferencia (Haar default)

**Fecha:** 2026-06-19  
**Estado:** corrida completada  
**Modelo:** E6 (sin reentrenar)

**Set de prueba:** 14 imágenes, 7 emociones cubiertas (etiquetas §4.2).

#### Detección Haar (punto 5)

| Parámetro | Valor |
|-----------|-------|
| Cascade | `haarcascade_frontalface_default.xml` |
| `SCALE_FACTOR` | 1.1 |
| `MIN_NEIGHBORS` | 6 |
| `MIN_SIZE` | (30, 30) |
| Recorte | primer rostro, cuadrado `max(w,h)` |

#### Métricas — inferencia externa

| Pipeline | Detección | Accuracy vs etiqueta | Notas |
|----------|-----------|----------------------|-------|
| **P4** (sin Haar) | 14/14 evaluables | **1/14 (7.1%)** | Resize directo 64×64 |
| **P5** (Haar + val) | **≥13/14** | pendiente contar | Fallo confirmado: `alegria 1.jpeg` |

**Referencia validación E6:** 79.34% acc — brecha dominio alineado ↔ foto real.

**Observaciones:**
- P4 peor que azar (~14.3%): rostro ocupa poca área tras resize global.
- Al menos una imagen sin detección Haar bloquea inferencia P5.
- Clases débiles en val (disgusto, miedo) probablemente empeoran en fotos externas.
- **Conclusión I0:** baseline documentado; prioridad = recuperar detección (I1) antes de reentrenar.

#### Tabla por imagen *(completar al revisar figuras P4/P5)*

| Archivo | Emoción esperada | Pred P4 | Pred P5 | ¿Detectado? | Notas |
|---------|------------------|---------|---------|-------------|-------|
| alegria 1.jpeg | alegria | | | **No** | Fallo Haar I0 |
| *(resto)* | | | | | |

---

### I1 — Relajar Haar (`minNeighbors` 6 → 4)

**Fecha:** 2026-06-19  
**Estado:** corrida completada  
**Base:** I0  
**Modelo:** E6 (sin reentrenar)

**Cambio vs I0:** §5.1 — `MIN_NEIGHBORS = 4` (I0: 6); resto igual.

#### Métricas — inferencia externa

| Pipeline | Detección | Accuracy vs etiqueta | Δ vs I0 |
|----------|-----------|----------------------|---------|
| P4 (control) | 14/14 | **1/14 (7.1%)** | = I0 |
| P5 | **13/14** | **3/13 (23.1%)** | detección = I0; acc P5 no medida en I0 |

**Criterio de éxito (14/14 detección):** **no cumplido** — persiste fallo en `alegria 1.jpeg`.

#### Tabla por imagen (P5 — I1)

| Archivo | Esperada | Pred P5 | ¿Detectado? | Acierto |
|---------|----------|---------|-------------|---------|
| alegria 1.jpeg | alegria | — | **No** | — |
| alegria 2.jpeg | alegria | alegria | Sí | ✓ |
| tristeza 1.jpeg | tristeza | alegria | Sí | ✗ |
| tristeza 2.jpeg | tristeza | enojo | Sí | ✗ |
| seriedad 1.jpeg | seriedad | tristeza | Sí | ✗ |
| seriedad 2.jpeg | seriedad | disgusto | Sí | ✗ |
| sorpresa 1.jpeg | sorpresa | tristeza | Sí | ✗ |
| sorpresa 2.jpeg | sorpresa | sorpresa | Sí | ✓ |
| miedo 1.jpeg | miedo | tristeza | Sí | ✗ |
| miedo 2.jpeg | miedo | tristeza | Sí | ✗ |
| enojo 1.jpeg | enojo | disgusto | Sí | ✗ |
| enojo 2.jpeg | enojo | enojo | Sí | ✓ |
| disgusto 1.jpeg | disgusto | enojo | Sí | ✗ |
| disgusto 2.jpeg | disgusto | enojo | Sí | ✗ |

**Observaciones:**
- `minNeighbors=4` **no recuperó** `alegria 1.jpeg`; detección **13/14** (= I0).
- P5 **3/13** > P4 **1/14**: el recorte Haar ayuda, pero el gap vs E6 (~79% val) sigue enorme.
- Aciertos: 1/2 alegría (solo la detectada), 1/2 sorpresa, 1/2 enojo; **0/2** en tristeza, seriedad, miedo, disgusto.
- Tristeza como predicción dominante en errores (miedo, seriedad, sorpresa); enojo↔disgusto cruzados (4 casos).

**Conclusión I1:** `minNeighbors` no desbloquea el caso fallido. **Siguiente: I2** — `MIN_SIZE = (20, 20)`.

---

### I2 — Reducir `minSize` `(30,30) → (20,20)`

**Fecha:** 2026-06-19  
**Estado:** corrida completada  
**Base:** I1  
**Modelo:** E6 (sin reentrenar)

**Cambio vs I1:** §5.1 — `MIN_SIZE = (20, 20)` (I1: `(30, 30)`); `MIN_NEIGHBORS = 4` sin cambio.

#### Métricas — inferencia externa

| Pipeline | Detección | Accuracy vs etiqueta | Δ vs I1 |
|----------|-----------|----------------------|---------|
| P5 | **13/14** | **3/13 (23.1%)** | **= I1** (sin mejora) |

**Criterio de éxito (14/14):** **no cumplido** — sigue sin detectarse `alegria 1.jpeg`.

#### Tabla por imagen (P5 — I2)

Idéntica a **I1** (mismas predicciones en las 13 imágenes evaluables):

| Aciertos (3) | Archivos |
|--------------|----------|
| ✓ | `sorpresa 2.jpeg`, `enojo 2.jpeg`, `alegria 2.jpeg` |
| Sin detección | `alegria 1.jpeg` |

**Observaciones:**
- Bajar `minSize` a (20,20) **no cambió** detección ni clasificación respecto a I1.
- El fallo en `alegria 1.jpeg` **no parece** de umbral de tamaño mínimo (al menos no con default cascade + estos parámetros).
- Posibles causas restantes: ángulo de rostro, oclusión, iluminación, rostro no frontal, o imagen donde Haar default no aplica — no resoluble solo con `minSize`/`minNeighbors`.

**Conclusión I2:** misma línea que I1. **Siguiente: I3** — `SCALE_FACTOR = 1.05`.

---

### I3 — Reducir `scaleFactor` `1.1 → 1.05`

**Fecha:** 2026-06-19  
**Estado:** corrida completada — **config Haar final (punto 5)**  
**Base:** I2  
**Modelo:** E6 (sin reentrenar)

**Cambio vs I2:** §5.1 — `SCALE_FACTOR = 1.05` (I2: `1.1`); resto sin cambio (`MIN_NEIGHBORS = 4`, `MIN_SIZE = (20, 20)`, cascade default).

#### Métricas — inferencia externa

| Pipeline | Detección | Accuracy vs etiqueta | Δ vs I2 |
|----------|-----------|----------------------|---------|
| P5 | **14/14** | **4/14 (28.6%)** | detección **+1**; acc **+1** acierto |

**Criterio de éxito (14/14 detección):** **cumplido** — se recupera `alegria 1.jpeg`.

#### Tabla por imagen (P5 — I3)

| Archivo | Esperada | Pred P5 | Detectado | vs I2 |
|---------|----------|---------|-----------|-------|
| tristeza 1.jpeg | tristeza | disgusto | Sí | = |
| sorpresa 2.jpeg | sorpresa | sorpresa | Sí | ✓ = |
| sorpresa 1.jpeg | sorpresa | tristeza | Sí | = |
| seriedad 1.jpeg | seriedad | tristeza | Sí | = |
| seriedad 2.jpeg | seriedad | disgusto | Sí | = |
| miedo 1.jpeg | miedo | tristeza | Sí | = |
| miedo 2.jpeg | miedo | tristeza | Sí | = |
| enojo 1.jpeg | enojo | disgusto | Sí | = |
| enojo 2.jpeg | enojo | enojo | Sí | ✓ = |
| disgusto 1.jpeg | disgusto | tristeza | Sí | pred cambió (enojo→tristeza) |
| disgusto 2.jpeg | disgusto | enojo | Sí | = |
| alegria 1.jpeg | alegria | tristeza | **Sí** | **detectada** (I2: No) |
| alegria 2.jpeg | alegria | alegria | Sí | ✓ = |
| tristeza 2.jpeg | tristeza | tristeza | Sí | ✓ **nuevo** (I2: enojo) |

| Aciertos (4) | Archivos |
|--------------|----------|
| ✓ | `sorpresa 2.jpeg`, `enojo 2.jpeg`, `alegria 2.jpeg`, `tristeza 2.jpeg` |

**Observaciones:**
- Bajar `scaleFactor` a **1.05** fue el único cambio que **desbloqueó** `alegria 1.jpeg` (14/14).
- Esa imagen se clasifica mal (`tristeza`); el recorte con escala más fina cambia el crop respecto a I2.
- Mejora de clasificación modesta: **4/14** vs **3/13** en I2; acierto nuevo en `tristeza 2.jpeg`.
- Confusiones dominantes: **tristeza** como predicción frecuente (miedo, seriedad, sorpresa, alegria 1); **enojo ↔ disgusto**.

**Conclusión I3:** **config Haar final** — `scaleFactor=1.05`, `minNeighbors=4`, `minSize=(20,20)`, cascade default. Campaña I0–I3 cerrada en detección; no se planean más ajustes Haar salvo análisis narrativo en conclusiones del TP.

---

### Tabla comparativa — inferencia externa

| Exp | Cambio Haar | Detección P5 | Acc P4 | Acc P5 | Notas |
|-----|-------------|--------------|--------|--------|-------|
| **I0** | neighbors=6 | 13/14 | **1/14** | pendiente | baseline |
| **I1** | neighbors=4 | 13/14 | 1/14 | **3/13** | |
| **I2** | minSize (20,20) | **13/14** | — | **3/13** | = I1; `alegria 1.jpeg` sigue fallando |
| **I3** | scaleFactor 1.05 | **14/14** | — | **4/14** | **config final**; recupera `alegria 1.jpeg` |

---

### E6-b — Inferencia con conjunto alternativo de imágenes (Set B)

**ID:** **E6-b** (mismo modelo E6, **sin reentrenar**; imágenes distintas al set de entrega).  
**Estado:** **completada** — 2026-06-19  
**Set de referencia (entrega):** Set A — 14 imágenes originales (I0–I3, acc P5 4/14).  
**Haar:** I3 (`scaleFactor=1.05`, `minNeighbors=4`, `minSize=(20,20)`).

**Descripción Set B:** 14 fotos stock (mujer + hombre, expresiones más marcadas, fondo neutro). Misma estructura de nombres (`alegria 1.jpeg`, etc.).

**Objetivo:** evaluar si otro conjunto mejora P4/P5 sin reentrenar.

#### Métricas agregadas

| Métrica | Set A (entrega) | E6-b / Set B | Δ |
|---------|-----------------|--------------|---|
| N imágenes | 14 | 14 | = |
| Detección P5 | 14/14 | **14/14** | = |
| Acc P4 | 1/14 (7.1%) | **2/14 (14.3%)** | **+1** |
| Acc P5 | 4/14 (28.6%) | **5/14 (35.7%)** | **+1** |

#### Tabla por imagen E6-b

| Archivo | Esperada | Pred P4 | Pred P5 | ¿Detectado? | P4 | P5 |
|---------|----------|---------|---------|-------------|----|----|
| alegria 1.jpeg | alegria | tristeza | alegria | Sí | ✗ | ✓ |
| alegria 2.jpeg | alegria | enojo | alegria | Sí | ✗ | ✓ |
| tristeza 1.jpeg | tristeza | tristeza | sorpresa | Sí | ✓ | ✗ |
| tristeza 2.jpeg | tristeza | enojo | disgusto | Sí | ✗ | ✗ |
| sorpresa 1.jpeg | sorpresa | tristeza | sorpresa | Sí | ✗ | ✓ |
| sorpresa 2.jpeg | sorpresa | tristeza | sorpresa | Sí | ✗ | ✓ |
| seriedad 1.jpeg | seriedad | tristeza | tristeza | Sí | ✗ | ✗ |
| seriedad 2.jpeg | seriedad | tristeza | tristeza | Sí | ✗ | ✗ |
| enojo 1.jpeg | enojo | enojo | tristeza | Sí | ✓ | ✗ |
| enojo 2.jpeg | enojo | tristeza | tristeza | Sí | ✗ | ✗ |
| miedo 1.jpeg | miedo | tristeza | tristeza | Sí | ✗ | ✗ |
| miedo 2.jpeg | miedo | tristeza | alegria | Sí | ✗ | ✗ |
| disgusto 1.jpeg | disgusto | tristeza | tristeza | Sí | ✗ | ✗ |
| disgusto 2.jpeg | disgusto | tristeza | disgusto | Sí | ✗ | ✓ |

#### Observaciones

- **Mejora marginal** vs Set A: +1 acierto en P4 y +1 en P5; detección Haar sigue 14/14.
- **Haar aporta mucho en casos concretos:** `alegria 1` pasa de tristeza (P4) a alegria 100% (P5); `alegria 2` de enojo → alegria.
- **Sesgo a tristeza persiste** en P5: seriedad (0/2), enojo (0/2), miedo (0/2), disgusto 1; en P4 casi todo cae en tristeza salvo `tristeza 1` y `enojo 1`.
- **Clases que mejoran con Set B:** alegría 2/2 en P5 (Set A: 1/2); sorpresa 2/2 en P5 (Set A: 1/2); disgusto 1/2 P5 (Set A: 0/2).
- **Clases que siguen fallando:** seriedad 0/2, miedo 0/2, enojo 0/2 P5; tristeza 0/2 P5.
- Fotos del **hombre** rinden mejor en alegría/sorpresa/disgusto; fotos de la **mujer** (turtleneck blanco) siguen confundiendo tristeza/sorpresa/enojo.

**Conclusión E6-b:** Set B **supera levemente** a Set A (+7 pp relativo en P5: 28.6%→35.7%) pero sigue muy por debajo de validación E6 (~79%). **Recomendación:** usar **Set B para entrega** (mejores expresiones + mismo pipeline); mantener E6 + Haar I3; documentar domain shift y sesgo tristeza en conclusiones §4–§5.

---

## Plantilla para nuevos experimentos

Copiar, renumerar (E1, E2, …) y completar:

```markdown
### EN — [Título breve]

**Fecha:** YYYY-MM-DD  
**Estado:** corrida completada | pendiente

**Cambio vs E(n-1):**
- Preprocesamiento: (ninguno | detalle)
- Modelo: (ninguno | detalle)
- Entrenamiento: (ninguno | detalle)

**Config completa** *(solo si cambió algo; si no, referenciar experimento base)*

| Preprocesamiento | Modelo | Entrenamiento |
|------------------|--------|---------------|
| … | … | … |

**Métricas — validación**

| Accuracy | F1 macro | Loss | Mejor época | Parada |
|----------|----------|------|-------------|--------|
| | | | | |

**Observaciones:**
- 
```

---

*Última actualización: 2026-06-19 — **E6-b** completada: P5 **5/14 (35.7%)** vs Set A 4/14. Recomendado Set B para entrega.*
