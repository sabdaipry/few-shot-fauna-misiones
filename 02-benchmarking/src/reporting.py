"""Generación del reporte HTML para el benchmark de fauna de la Selva Paranaense."""
import pandas as pd
from pathlib import Path
from .visualization import standardize_columns, filter_data

def get_custom_sort_key(item):
    """
    Define el orden de visualización de los UMAPs.
    Orden: ResNet -> ConvNext -> DINO -> BioCLIP -> SigLIP -> CLIP
    """
    model_name = item[0]
    # Prioridades (Menor número = aparece antes)
    if 'resnet' in model_name:   return (6, model_name)
    if 'convnext' in model_name: return (1, model_name)
    if 'dino' in model_name:     return (2, model_name)
    if 'bioclip' in model_name:  return (3, model_name)
    if 'siglip' in model_name:   return (4, model_name)
    if 'clip' in model_name:     return (5, model_name)
    return (99, model_name) # Otros al final

def generate_html_report(data, output_file, umap_files, backbones_list, classifiers_list, stats):
    """Genera el dashboard HTML."""
    
    table_class = "table table-striped table-hover table-bordered table-sm"
    
    # Preparar Datos
    df_raw = standardize_columns(data['summary'])
    df_clean = filter_data(df_raw)
    # Ordenar por Accuracy (para el Top 10)
    df_top = df_clean.sort_values("Accuracy", ascending=False).head(10)
    
    # --- HTML COMPONENTS ---

    # 1. Top 10 Cards
    top_cards_html = ""
    for i, (_, row) in enumerate(df_top.iterrows()):
        # --- CAMBIO 3: LÓGICA DE RESALTADO ---
        if i == 0:
            # Estilo para el ganador (Verde clarito + Borde fuerte)
            card_style = "background-color: #e8f5e9; border: 2px solid #2ecc71;"
            badge = f'<div class="badge bg-success mb-2">🏆 #{i+1}</div>'
            text_class = "text-success"
        else:
            # Estilo normal
            card_style = "background-color: white;"
            badge = f'<div class="badge bg-secondary mb-2" style="opacity: 0.5">#{i+1}</div>'
            text_class = "text-muted"

        # --- CAMBIO 2: GRID RESPONSIVA (col-xl-2 para pantallas grandes, col-md-4 para medianas) ---
        top_cards_html += f"""
        <div class="col-xl-2 col-lg-3 col-md-4 col-sm-6">
            <div class="metric-box h-100" style="{card_style}">
                {badge}
                <h6 class="text-truncate fw-bold {text_class}" title="{row['Embedding Model']}">{row['Embedding Model']}</h6>
                <p class="small mb-1 text-truncate" title="{row.get('Classifier', '')}">{row.get('Classifier', '')}</p>
                <div class="metric-val my-2" style="font-size: 1.6rem;">{row.get('Accuracy', 0):.4f}</div>
                <div class="metric-lbl">Accuracy</div>
            </div>
        </div>
        """
    
    # 2. Dataset Stats HTML
    stats_html = f"""
    <div class="row mb-5">
        <div class="col-md-3">
            <div class="metric-box border-start border-4 color-forest">
                <div class="metric-val color-forest" style="font-size: 1.8rem;"><strong>{stats.get('total_images', 0)}</strong></div>
                <div class="metric-lbl">Imágenes Totales</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="metric-box border-start border-4 color-leaf">
                <div class="metric-val color-leaf" style="font-size: 1.8rem;"><strong>{stats.get('total_species', 0)}</strong></div>
                <div class="metric-lbl">Especies (Clases)</div>
                <small class="text-muted">({stats.get('total_families', 0)} Familias)</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="metric-box border-start border-4 color-gold">
                <div class="metric-val color-gold" style="font-size: 1.8rem;"><strong>{stats.get('gallery_count', 0)}</strong></div>
                <div class="metric-lbl">Set de Galería</div>
                <small class="text-muted">Referencias</small>
            </div>
        </div>
        <div class="col-md-3">
            <div class="metric-box border-start border-4 color-sand">
                <div class="metric-val color-sand" style="font-size: 1.8rem;"><strong>{stats.get('query_count', 0)}</strong></div>
                <div class="metric-lbl">Set de <i>Query</i></div>
                <small class="text-muted">Validación</small>
            </div>
        </div>
    </div>
    """

    # 3. Opciones para Dropdowns
    bb_options = "".join([f'<option value="{b}">{b}</option>' for b in backbones_list])
    # Para los clasificadores, ponemos 'Nearest Centroid' primero por defecto si existe
    sorted_clfs = sorted(classifiers_list, key=lambda x: (x != 'Nearest Centroid', x))
    clf_options = "".join([f'<option value="{c}">{c}</option>' for c in sorted_clfs])

    # 4. Tablas 
    table_sys_html = "<p>No data</p>"
    if 'incremental' in data and not data['incremental'].empty:
        df_inc = data['incremental']
        cols_inc = [c for c in df_inc.columns if "Time" in c or "Tiempo" in c or "Model" in c]
        table_sys_html = df_inc[cols_inc].round(4).to_html(classes=table_class, index=False, border=0)

    table_out_html = "<p>No data</p>"
    if 'outliers' in data and not data['outliers'].empty:
        df_out = data['outliers']
        cols_out = [c for c in df_out.columns if c != "Threshold Distance"] 
        table_out_html = df_out[cols_out].round(4).to_html(classes=table_class, index=False, border=0)
        
    # Tabla Completa (Aquí sí mostramos todo, o filtramos? Mejor filtrar FAISS por limpieza)
    df_full_display = df_clean.sort_values("Accuracy", ascending=False)
    # cols_show = [c for c in ['Embedding Model', 'Classifier', 'Accuracy', 'F1_Macro', 'Dim', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'] if c in df_full_display.columns]
    table_main_html = df_full_display.round(4).to_html(classes=table_class, index=False, border=0)

    # 5. Galería UMAP (ordenada)
    sorted_umaps = sorted(umap_files.items(), key=get_custom_sort_key)

    umap_gallery_html = '<div class="row">'
    for model_name, img_path in sorted_umaps:
        rel_path = f"figures/umaps/{Path(img_path).name}"
        umap_gallery_html += f"""
        <div class="col-md-6 mb-4">
            <div class="card h-100 border-0 shadow-sm">
                <img src="{rel_path}" class="card-img-top" alt="{model_name}" loading="lazy">
                <div class="card-footer bg-white text-center fw-bold small">{model_name}</div>
            </div>
        </div>
        """
    umap_gallery_html += '</div>'


    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Benchmark Fauna Selva Paranaense - Reporte Final</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f0f4f1; font-family: 'Segoe UI', sans-serif; padding-bottom: 60px; }}
            .header {{ background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); color: white; padding: 3rem 0; margin-bottom: 2rem; }}
            .section-title {{ border-left: 6px solid #2ecc71; padding-left: 15px; margin: 50px 0 25px 0; color: #1e5128; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; }}
            .card {{ margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.06); border: none; border-radius: 12px; }}
            .metric-box {{ text-align: center; padding: 1.5rem; background: white; border-radius: 12px; box-shadow: 0 3px 5px rgba(0,0,0,0.05); height: 100%; }}
            .metric-box.highlight {{ border-top: 5px solid #2ecc71; }}
            .nav-tabs .nav-link.active {{ background-color: #e8f5e9; border-bottom: 3px solid #27ae60; font-weight: bold; color: #1e5128; }}
            .nav-tabs .nav-link {{ color: #555; }}
            .color-forest {{ border-color: #2d4c1e !important; color: #2d4c1e !important; }}
            .color-leaf   {{ border-color: #7fb439 !important; color: #7fb439 !important; }}
            .color-gold   {{ border-color: #c0ad5d !important; color: #c0ad5d !important; }}
            .color-sand   {{ border-color: #d9b373 !important; color: #d9b373 !important; }}
            img {{ border-radius: 8px; }}
        </style>
    </head>
    <body>
    
    <div class="header text-center">
        <img src="assets/logo.png" alt="Logo Proyecto" class="logo-img" style="max-height: 200px;" onerror="this.style.display='none'">
        <h1>Reporte Técnico Final</h1>
        <p class="lead">Benchmark de Modelos para Clasificación de Fauna de la Selva Paranaense</p>
    </div>

    <div class="container">

        <h4 class="text-muted mb-3">Resumen del Dataset Evaluado</h4>
        {stats_html}
        
        <h3 class="section-title">🏆 Top 10 Combinaciones (Accuracy)</h3>
        <div class="row mb-4 justify-content-center g-3">{top_cards_html}</div>

        <h3 class="section-title">⚔️ Comparación de Desempeño Taxonómico</h3>

        <div class="card mb-4">
            <div class="card-header fw-bold bg-white">Guía de Interpretación de Métricas Biológicas</div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6 border-end">
                        <h6 class="text-success fw-bold">Índice de Valor de Conservación (IVC)</h6>
                        <p class="small text-muted mb-2">
                            Métrica que integra riesgo biológico y funcional.
                        </p>
                        <p class="small text-muted mb-2">
                            Fórmula: <code>Wi = S_cons + F_end + F_eco</code>
                        </p>
                        <ul class="small mb-0">
                            <li><strong>Urgencia (<code>S_cons</code>):</strong> Escala 1-5 según riesgo de extinción (IUCN/SAREM). Penaliza especies en Peligro Crítico.</li>
                            <li><strong>Irremplazabilidad (<code>F_end</code>):</strong> (+2) Endémicas del Bosque Atlántico.</li>
                            <li><strong>Funcionalidad (<code>F_eco</code>):</strong> (+1) Ingenieros ecosistémicos o depredadores tope.</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6 class="text-danger fw-bold">Gravedad de Errores Taxonómicos</h6>
                        <ul class="small mb-0">
                            <li><span class="badge bg-warning text-dark">Leve</span> <strong>Género Correcto:</strong> Confunde especies hermanas (ej: <em>Turdus rufiventris</em> vs <em>Turdus leucomelas</em>).</li>
                            <li><span class="badge bg-orange text-white" style="background-color: #fd7e14;">Medio</span> <strong>Familia Correcta:</strong> Confunde géneros dentro del mismo grupo (ej: Felinos distintos).</li>
                            <li><span class="badge bg-danger">Severo</span> <strong>Familia Incorrecta:</strong> Confunde grupos biológicos distintos (ej: Ave con Mamífero).</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <div class="card bg-light">
            <div class="card-body">
                <p class="text-center text-muted mb-4">Configura los contendientes para analizar sus desempeños taxonómicos.</p>
                
                <div class="row">
                    <div class="col-md-6 border-end">
                        <h5 class="text-center mb-3">Modelo A</h5>
                        <div class="row g-2 mb-3">
                            <div class="col-md-6">
                                <label class="small text-muted">Backbone</label>
                                <select class="form-select arena-select" id="bb1" onchange="updateArena('1')">
                                    <option value="" disabled selected>Elegir...</option>
                                    {bb_options}
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="small text-muted">Clasificador</label>
                                <select class="form-select arena-select" id="clf1" onchange="updateArena('1')">
                                    {clf_options}
                                </select>
                            </div>
                        </div>
                        
                        <div class="card mb-3">
                            <div class="card-header fw-bold text-center small">Errores Taxonómicos</div>
                            <div class="card-body p-1 text-center">
                                <img id="imgTax1" src="figures/placeholder.png" style="max-height: 300px;" onerror="this.style.display='none'">
                                <p id="errTax1" class="text-danger small mt-2" style="display:none">Gráfico no disponible</p>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-header fw-bold text-center small">Desempeño IVC</div>
                            <div class="card-body p-1 text-center">
                                <img id="imgIVC1" src="figures/placeholder.png" style="max-height: 300px;" onerror="this.style.display='none'">
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <h5 class="text-center mb-3">Modelo B</h5>
                        <div class="row g-2 mb-3">
                            <div class="col-md-6">
                                <label class="small text-muted">Backbone</label>
                                <select class="form-select arena-select" id="bb2" onchange="updateArena('2')">
                                    <option value="" disabled selected>Elegir...</option>
                                    {bb_options}
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="small text-muted">Clasificador</label>
                                <select class="form-select arena-select" id="clf2" onchange="updateArena('2')">
                                    {clf_options}
                                </select>
                            </div>
                        </div>
                        
                        <div class="card mb-3">
                            <div class="card-header fw-bold text-center small">Errores Taxonómicos</div>
                            <div class="card-body p-1 text-center">
                                <img id="imgTax2" src="figures/placeholder.png" style="max-height: 300px;" onerror="this.style.display='none'">
                                <p id="errTax2" class="text-danger small mt-2" style="display:none">Gráfico no disponible</p>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-header fw-bold text-center small">Desempeño IVC</div>
                            <div class="card-body p-1 text-center">
                                <img id="imgIVC2" src="figures/placeholder.png" style="max-height: 300px;" onerror="this.style.display='none'">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <h3 class="section-title">Rendimiento General (Accuracy)</h3>
        <div class="card"><div class="card-body">
            <ul class="nav nav-tabs" id="perfTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="rank-tab" data-bs-toggle="tab" data-bs-target="#rank-pane" type="button" role="tab">Ranking</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="heat-tab" data-bs-toggle="tab" data-bs-target="#heat-pane" type="button" role="tab">Mapa de Calor</button>
                </li>
            </ul>
            <div class="tab-content p-3" id="perfTabContent">
                <div class="tab-pane fade show active" id="rank-pane" role="tabpanel">
                    <img src="figures/01_leaderboard_acc.png" class="img-fluid d-block mx-auto" style="max-height: 800px;">
                </div>
                <div class="tab-pane fade" id="heat-pane" role="tabpanel">
                    <img src="figures/04_heatmap_acc.png" class="img-fluid d-block mx-auto" style="max-height: 800px;">
                </div>
            </div>
        </div></div>


        <h3 class="section-title">Eficiencia</h3>
        <div class="row">
            <div class="col-md-12">
                <div class="card"><div class="card-body text-center">
                    <img src="figures/02_pareto_acc.png" class="img-fluid" style="max-height: 700px;">
                </div></div>
            </div>
        </div>
        <div class="row">
            <div class="col-md-6">
                 <div class="card"><div class="card-body text-center">
                    <img src="figures/latency_backbone.png" class="img-fluid">
                </div></div>
            </div>
            <div class="col-md-6">
                 <div class="card"><div class="card-body text-center">
                    <img src="figures/latency_classifier.png" class="img-fluid">
                </div></div>
            </div>
        </div>

        <h3 class="section-title">Calidad de Embeddings</h3>
        <div class="card"><div class="card-body">
            <div class="row">
                <div class="col-md-12 mb-4"><img src="figures/emb_metric_silhouette_score.png" class="img-fluid border"></div>
                <div class="col-md-12 mb-4"><img src="figures/emb_metric_davies-bouldin_index.png" class="img-fluid border"></div>
                <div class="col-md-12"><img src="figures/emb_metric_calinski-harabasz_index.png" class="img-fluid border"></div>
            </div>
        </div></div>
        
        <h3 class="section-title">Escalabilidad y Sistema</h3>
        <div class="card"><div class="card-body text-center">
             <img src="figures/03_scalability.png" class="img-fluid mb-4 border">
             <div class="row text-start">
                 <div class="col-md-6">
                     <h6>Tiempos de Registro Incremental</h6>
                     <div class="table-responsive">{table_sys_html}</div>
                 </div>
                 <div class="col-md-6">
                     <h6>Detección de Outliers</h6>
                     <div class="table-responsive">{table_out_html}</div>
                 </div>
             </div>
        </div></div>
        
        <h3 class="section-title">Análisis DINO (GAP vs CLS)</h3>
        <div class="info-box">
            <strong>Justificación Metodológica:</strong>
            <p style="text-align: justify;"> Se evalúa el impacto de la estrategia de "<i>pooling</i>" en arquitecturas ViT (<i>Vision Transformers</i>). 
            Se compara el uso del <i>token</i> especial <code>[CLS]</code> (entrenado para capturar la semántica global de la clase) 
            versus el <code>GAP</code> (<i>Global Average Pooling</i>), que promedia los vectores de características espaciales. 
            Esto permite determinar si la información discriminativa reside en el <i>token</i> de clasificación o distribuida en los parches visuales.
            </p>
        </div>
        <div class="card"><div class="card-body text-center">
            <img src="figures/05_dino_comparison_acc.png" class="img-fluid" style="max-width: 800px;">
        </div></div>

        <h3 class="section-title">Galería UMAP</h3>
        {umap_gallery_html}

        <h3 class="section-title">Resultados Completos</h3>
        <div class="card"><div class="card-body">
            <div class="table-responsive" style="max-height: 600px; overflow-y: auto;">
            {table_main_html}
            </div> 
        </div></div>
        
    </div>
    <footer class="text-center py-5 text-muted small">S. Pryszczuk | 2026</footer>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function cleanFilename(name) {{
            // Replica la lógica de Python: lowercase, spaces->underscore, remove parenthesis
            return name.toLowerCase()
                       .replace(/ \\(/g, '_')
                       .replace(/\\)/g, '')
                       .replace(/=/g, '')
                       .replace(/ /g, '_');
        }}

        function updateArena(side) {{
            const bb = document.getElementById('bb' + side).value;
            const clf = document.getElementById('clf' + side).value;
            
            if (!bb || !clf) return;
            
            const clfSlug = cleanFilename(clf);
            
            // Construir rutas
            const taxPath = `figures/errors/tax_error_${{bb}}_${{clfSlug}}.png`;
            const ivcPath = `figures/ivc/ivc_perf_${{bb}}_${{clfSlug}}.png`;
            
            const imgTax = document.getElementById('imgTax' + side);
            const imgIVC = document.getElementById('imgIVC' + side);
            
            // Reset display y asignar src
            imgTax.style.display = 'block';
            imgIVC.style.display = 'block';
            document.getElementById('errTax' + side).style.display = 'none';
            
            imgTax.src = taxPath;
            imgIVC.src = ivcPath;
            
            // Manejador de error simple
            imgTax.onerror = function() {{
                this.style.display = 'none';
                document.getElementById('errTax' + side).style.display = 'block';
                document.getElementById('errTax' + side).innerText = "No data for " + bb + " + " + clf;
            }};
        }}
        
        // Inicializar selects
        window.onload = function() {{
            // Opcional: Seleccionar valores por defecto
        }};
    </script>
    </body>
    </html>
    """
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)