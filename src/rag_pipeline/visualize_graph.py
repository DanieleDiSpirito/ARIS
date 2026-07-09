import os
import re
import json
import argparse
import networkx as nx
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

COLLECTION_NAME = "langchain"

def get_db_path(env: str, chunk_size: int, metodo: str = "pdf4llm") -> str:
    return os.path.join("vector_db", f"chroma_{metodo}_{env}_{chunk_size}")

def get_embeddings(env: str):
    if env == "locale":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif env == "cloud":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"env non valido: '{env}'.")

# Importiamo la stessa logica di estrazione entità di LightweightGraphRAG
def extract_entities(text: str) -> list:
    entities = []
    error_codes = re.findall(r'\b[A-Z]{4}-\d{3}\b', text)
    entities.extend([code.upper() for code in error_codes])
    part_numbers = re.findall(r'\bA\d{2}B-\d{4}-\d{4}\b|\bA\d{2}B-\d{3}-\d{4}\b|\bA\d{2}B-\d{4}-[A-Z]\d{3}\b|\bA05B-\d{4}-[A-Z]\d{3}\b|\bA06B-\d{4}-[A-Z]\d{3}\b', text)
    entities.extend([pn.upper() for pn in part_numbers])
    connectors = re.findall(r'\b(?:CRMA\d{2}[A-Z]?|JD\d{1}[A-Z]|CNJ[xXyYzZ\d]+|CRR\d{2})\b', text)
    entities.extend([conn.upper() for conn in connectors])
    return list(set(entities))

def _get_node_key(doc: Document) -> str:
    import hashlib
    chunk_id = doc.metadata.get("chunk_id", "unknown")
    text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]
    return f"{chunk_id}_{text_hash}"

def main():
    parser = argparse.ArgumentParser(description="Genera una visualizzazione interattiva HTML del Grafo GraphRAG")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale")
    parser.add_argument("--chunk_size", type=int, default=700)
    parser.add_argument("--metodo", type=str, default="pdf4llm")
    parser.add_argument("--output", type=str, default=os.path.join("knowledge_graphs", "graph.html"), help="File HTML di output")
    parser.add_argument("--theme", type=str, choices=["dark", "light"], default="dark", help="Tema grafico del file HTML (default: dark)")
    
    # Filtri
    parser.add_argument("--keyword", type=str, help="Filtra i nodi contenenti questa parola chiave e i loro vicini")
    parser.add_argument("--file", type=str, help="Mostra solo nodi associati a questo file PDF")
    parser.add_argument("--min-weight", type=float, default=1.0, help="Peso minimo del collegamento da visualizzare (default: 1.0)")
    args = parser.parse_args()

    # Imposta la root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    os.chdir(project_root)

    db_path = get_db_path(args.env, args.chunk_size, args.metodo)
    if not os.path.exists(db_path):
        print(f"❌ Database non trovato: {db_path}")
        return

    embeddings = get_embeddings(args.env)
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    all_data = db.get()
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
        if text
    ]

    print(f"🕸️ Costruzione Grafo con {len(docs)} chunk...")
    G = nx.Graph()
    chunk_dict = {}

    for doc in docs:
        node_key = _get_node_key(doc)
        entities = extract_entities(doc.page_content)
        doc.metadata["entities"] = entities
        G.add_node(node_key, doc=doc)
        chunk_dict[node_key] = doc

    # Raggruppa i nodi per entità per creare archi
    entity_to_chunks = {}
    for doc in docs:
        node_key = _get_node_key(doc)
        for ent in doc.metadata.get("entities", []):
            entity_to_chunks.setdefault(ent, []).append(node_key)

    # Archi per entità condivise
    for ent, node_keys in entity_to_chunks.items():
        if len(node_keys) > 1:
            for i in range(len(node_keys)):
                for j in range(i + 1, len(node_keys)):
                    c1, c2 = node_keys[i], node_keys[j]
                    if G.has_edge(c1, c2):
                        G[c1][c2]["weight"] += 1.0
                        if ent not in G[c1][c2]["entities"]:
                            G[c1][c2]["entities"].append(ent)
                    else:
                        G.add_edge(c1, c2, weight=1.0, entities=[ent])

    # Archi per adiacenza sequenziale (stesso file, pagina adiacente)
    file_groups = {}
    for doc in docs:
        file_name = doc.metadata.get("file_name")
        if file_name:
            file_groups.setdefault(file_name, []).append(doc)

    for file_name, file_docs in file_groups.items():
        def get_page_num(d):
            p = d.metadata.get("page", "0")
            m = re.search(r'\d+', str(p))
            return int(m.group()) if m else 0
        sorted_docs = sorted(file_docs, key=get_page_num)
        for i in range(len(sorted_docs) - 1):
            d1 = sorted_docs[i]
            d2 = sorted_docs[i+1]
            p1 = get_page_num(d1)
            p2 = get_page_num(d2)
            if abs(p1 - p2) <= 1:
                c1 = _get_node_key(d1)
                c2 = _get_node_key(d2)
                if G.has_edge(c1, c2):
                    G[c1][c2]["weight"] += 0.5
                else:
                    G.add_edge(c1, c2, weight=0.5, entities=["adjacency"])

    print(f"✅ Grafo costruito: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi.")

    # Applicazione dei filtri sui nodi/archi
    nodes_to_keep = set(G.nodes())

    if args.file:
        target = args.file.lower()
        nodes_to_keep = {n for n in nodes_to_keep if target in chunk_dict[n].metadata.get("file_name", "").lower()}
        print(f"🔍 Filtrato per file '{args.file}': mantenuti {len(nodes_to_keep)} nodi.")

    if args.keyword:
        target = args.keyword.lower()
        matching = {n for n in G.nodes() if target in chunk_dict[n].page_content.lower()}
        print(f"🔍 Trovati {len(matching)} nodi contenenti '{args.keyword}'.")
        # Includiamo i nodi corrispondenti e tutti i loro vicini di primo grado
        neighbors = set()
        for n in matching:
            neighbors.update(G.neighbors(n))
        nodes_to_keep = nodes_to_keep.intersection(matching.union(neighbors))
        print(f"🔍 Includendo i vicini di primo grado: mantenuti {len(nodes_to_keep)} nodi.")

    # Costruiamo il sottografo
    subG = G.subgraph(nodes_to_keep).copy()

    # Rimuoviamo gli archi con peso inferiore alla soglia
    edges_to_remove = [(u, v) for u, v, d in subG.edges(data=True) if d.get("weight", 1.0) < args.min_weight]
    subG.remove_edges_from(edges_to_remove)

    # Rimuoviamo i nodi isolati risultanti (opzionale, ma mantiene la visualizzazione pulita)
    isolated_nodes = list(nx.isolates(subG))
    subG.remove_nodes_from(isolated_nodes)

    print(f"📊 Sottografo finale da visualizzare: {subG.number_of_nodes()} nodi, {subG.number_of_edges()} archi.")

    # Generazione dei dati JSON per vis.js
    vis_nodes = []
    vis_edges = []

    # Palette colori per file
    files_seen = list(set(chunk_dict[n].metadata.get("file_name", "unknown") for n in subG.nodes()))
    import colorsys
    def get_color(idx, total):
        if total == 0: return "#97c2fc"
        hue = idx / total
        r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.7)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    
    color_map = {f: get_color(idx, len(files_seen)) for idx, f in enumerate(files_seen)}

    for n in subG.nodes():
        doc = chunk_dict[n]
        f = doc.metadata.get("file_name", "unknown")
        p = doc.metadata.get("page", "N/A")
        label = f"Pag: {p} ({doc.metadata.get('chunk_id', 'N/A')})"
        vis_nodes.append({
            "id": n,
            "label": label,
            "title": f"File: {f} | Pagina: {p}",
            "color": color_map.get(f, "#97c2fc"),
            "file": f,
            "page": p,
            "entities": doc.metadata.get("entities", []),
            "text": doc.page_content
        })

    for u, v, d in subG.edges(data=True):
        weight = d.get("weight", 1.0)
        entities = d.get("entities", [])
        vis_edges.append({
            "from": u,
            "to": v,
            "value": weight,
            "title": f"Peso: {weight} | Entità: {', '.join(entities)}",
            "label": str(weight) if weight > 1.0 else ""
        })

    # Definizione delle variabili CSS del tema
    if args.theme == "light":
        css_vars = """
        :root {
            --bg-body: #f5f5f5;
            --text-body: #212121;
            --title-color: #0288d1;
            --bg-network: #ffffff;
            --border-color: #ccc;
            --bg-details: #ffffff;
            --bg-legend: #eaeaea;
            --border-legend: #ccc;
            --bg-tag: #e0e0e0;
            --text-tag: #212121;
            --node-font-color: #212121;
            --edge-color: #b0bec5;
            --node-border: #bdbdbd;
        }
        """
        vis_font_color = "#212121"
        vis_node_border = "#bdbdbd"
        vis_edge_color = "#b0bec5"
    else:
        css_vars = """
        :root {
            --bg-body: #121212;
            --text-body: #ffffff;
            --title-color: #4fc3f7;
            --bg-network: #1e1e1e;
            --border-color: #333;
            --bg-details: #1c1c1c;
            --bg-legend: #1a1a1a;
            --border-legend: #2d2d2d;
            --bg-tag: #333;
            --text-tag: #ffffff;
            --node-font-color: #ffffff;
            --edge-color: #555555;
            --node-border: #424242;
        }
        """
        vis_font_color = "#ffffff"
        vis_node_border = "#424242"
        vis_edge_color = "#555555"

    # Template HTML con CSS Variables
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Visualizzazione Grafo GraphRAG</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        CSS_VARS_PLACEHOLDER
        
        body {
            color: var(--text-body);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-body);
            margin: 0;
            padding: 20px;
        }
        h2 {
            margin-top: 0;
            color: var(--title-color);
        }
        .container {
            display: flex;
            flex-direction: row;
            height: 820px;
        }
        #mynetwork {
            flex-grow: 1;
            height: 100%;
            border: 1px solid var(--border-color);
            background-color: var(--bg-network);
            border-radius: 8px;
        }
        #details {
            width: 380px;
            padding: 20px;
            background-color: var(--bg-details);
            margin-left: 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            height: calc(100% - 40px);
            overflow-y: auto;
        }
        .legend-item {
            display: inline-block;
            margin-right: 15px;
            font-size: 12px;
        }
        .legend-color {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            vertical-align: middle;
            margin-right: 5px;
        }
        #legend {
            margin-bottom: 10px;
            padding: 10px;
            background-color: var(--bg-legend);
            border-radius: 6px;
            border: 1px solid var(--border-legend);
        }
        .btn {
            background-color: var(--title-color);
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            font-size: 13px;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 15px;
            font-weight: bold;
            transition: opacity 0.2s;
        }
        .btn:hover {
            opacity: 0.9;
        }
        #header-area {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        @media print {
            html, body {
                height: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            body {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            #details, h2, #header-area, .btn {
                display: none !important;
            }
            #legend {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ccc !important;
                margin: 5px 0 10px 0 !important;
                padding: 8px !important;
            }
            .container {
                height: calc(100% - 60px) !important;
                width: 100% !important;
                margin: 0 !important;
            }
            #mynetwork {
                border: none !important;
                width: 100% !important;
                height: 100% !important;
                background-color: #ffffff !important;
            }
        }
    </style>
</head>
<body>
    <div id="header-area">
        <h2 style="margin: 0;">Mappa Relazionale GraphRAG (ARIS)</h2>
        <div>
            <button class="btn" onclick="exportPNG()">📷 Esporta PNG Alta Risoluzione</button>
            <button class="btn" onclick="window.print()">🖨️ Stampa in PDF</button>
        </div>
    </div>
    <div id="legend">
        <strong>Legenda File: </strong>
        LEGEND_PLACEHOLDER
    </div>
    <div class="container">
        <div id="mynetwork"></div>
        <div id="details">
            <h3>Dettagli Nodo</h3>
            <p style="color: #888;">Seleziona un nodo del grafo per visualizzarne il contenuto e le relazioni.</p>
        </div>
    </div>

    <script type="text/javascript">
        var nodes = new vis.DataSet(NODES_PLACEHOLDER);
        var edges = new vis.DataSet(EDGES_PLACEHOLDER);

        var container = document.getElementById('mynetwork');
        var data = {
            nodes: nodes,
            edges: edges
        };
        var options = {
            nodes: {
                shape: 'dot',
                size: 16,
                font: {
                    color: 'VIS_FONT_COLOR_PLACEHOLDER',
                    size: 11
                },
                borderWidth: 1.5,
                borderColor: 'VIS_NODE_BORDER_PLACEHOLDER'
            },
            edges: {
                color: {
                    color: 'VIS_EDGE_COLOR_PLACEHOLDER',
                    highlight: '#ffcc00',
                    hover: '#ffcc00'
                },
                scaling: {
                    min: 1,
                    max: 5
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200
            },
            physics: {
                stabilization: true,
                barnesHut: {
                    gravitationalConstant: -10000,
                    springConstant: 0.03,
                    springLength: 100
                }
            }
        };
        var network = new vis.Network(container, data, options);

        network.on("click", function (params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var nodeData = nodes.get(nodeId);
                var detailDiv = document.getElementById('details');
                
                var html = '<h3 style="color: var(--title-color); margin-top: 0;">' + nodeData.label + '</h3>';
                html += '<p><strong>File:</strong> ' + nodeData.file + '</p>';
                html += '<p><strong>Pagina:</strong> ' + nodeData.page + '</p>';
                
                if (nodeData.entities.length > 0) {
                    html += '<p><strong>Entità individuate:</strong><br/>';
                    nodeData.entities.forEach(function(ent) {
                        html += '<span style="display:inline-block; background-color: var(--bg-tag); color: var(--text-tag); padding:2px 6px; margin:2px; border-radius:4px; font-size:11px;">' + ent + '</span>';
                    });
                    html += '</p>';
                }
                
                html += '<hr style="border: 0; border-top: 1px solid var(--border-color); margin: 15px 0;"/>';
                html += '<p style="white-space: pre-wrap; font-size: 13px; color: var(--text-body); line-height: 1.5; font-family: monospace;">' + nodeData.text + '</p>';
                
                detailDiv.innerHTML = html;
            }
        });

        function exportPNG() {
            var canvas = document.getElementsByTagName('canvas')[0];
            if (canvas) {
                var ctx = canvas.getContext('2d');
                ctx.save();
                
                var legendItems = document.querySelectorAll('.legend-item');
                
                var startX = 20;
                var startY = 30;
                var boxWidth = 330;
                var boxHeight = 25 + (legendItems.length * 20);
                
                // Disegna sfondo box legenda
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-details').trim() || '#ffffff';
                ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim() || '#cccccc';
                ctx.lineWidth = 1.5;
                
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(startX, startY, boxWidth, boxHeight, 8);
                } else {
                    ctx.rect(startX, startY, boxWidth, boxHeight);
                }
                ctx.fill();
                ctx.stroke();
                
                // Titolo legenda
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-body').trim() || '#000000';
                ctx.font = 'bold 12px "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
                ctx.fillText('LEGENDA FILE PDF', startX + 15, startY + 20);
                
                // Disegna le voci
                ctx.font = '11px "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
                legendItems.forEach(function(item, idx) {
                    var colorSpan = item.querySelector('.legend-color');
                    var color = colorSpan ? colorSpan.style.backgroundColor : '#97c2fc';
                    var text = item.textContent.trim();
                    
                    var itemY = startY + 45 + (idx * 20);
                    
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(startX + 22, itemY - 4, 6, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-body').trim() || '#000000';
                    ctx.fillText(text, startX + 38, itemY);
                });
                
                var link = document.createElement('a');
                link.download = 'mappa_relazionale_aris.png';
                link.href = canvas.toDataURL('image/png', 1.0);
                link.click();
                
                ctx.restore();
                network.redraw();
            }
        }
    </script>
</body>
</html>
"""

    # Crea la legenda HTML
    legend_html = ""
    for f, color in color_map.items():
        legend_html += f'<span class="legend-item"><span class="legend-color" style="background-color: {color};"></span>{f}</span>'

    # Rimpiazza i placeholder nel template
    html_content = html_template.replace("CSS_VARS_PLACEHOLDER", css_vars)
    html_content = html_content.replace("VIS_FONT_COLOR_PLACEHOLDER", vis_font_color)
    html_content = html_content.replace("VIS_NODE_BORDER_PLACEHOLDER", vis_node_border)
    html_content = html_content.replace("VIS_EDGE_COLOR_PLACEHOLDER", vis_edge_color)
    html_content = html_content.replace("NODES_PLACEHOLDER", json.dumps(vis_nodes, indent=2))
    html_content = html_content.replace("EDGES_PLACEHOLDER", json.dumps(vis_edges, indent=2))
    html_content = html_content.replace("LEGEND_PLACEHOLDER", legend_html)

    # Salva il file HTML
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n🎉 Visualizzazione salvata con successo in: [graph.html](file://{os.path.abspath(output_path)})")
    print("👉 Apri questo file nel tuo browser web per interagire con il grafo!")

if __name__ == "__main__":
    main()
