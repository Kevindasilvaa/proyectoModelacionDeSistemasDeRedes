import sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter
import heapq 
import json # <--- 1. Importar la librería JSON

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

# --- 2. Implementación Manual de Dijkstra (Sin cambios) ---
def find_shortest_path_dijkstra(graph, start, end):
    """
    Implementación manual del algoritmo de Dijkstra.
    Usa un grafo DiGraph de NetworkX como entrada.
    Devuelve (path, cost)
    """
    distances = {node: float('inf') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    
    if start not in distances:
        return [], float('inf')
        
    distances[start] = 0
    
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
            
        if current_node == end:
            break

        for neighbor in graph.neighbors(current_node):
            if not graph.has_edge(current_node, neighbor):
                continue
                
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
                        
    if distances[end] == float('inf'):
        return [], float('inf')
        
    path = []
    current = end
    while current != start:
        if current is None:
            return [], float('inf')
        path.append(current)
        current = predecessors.get(current)
    
    path.append(start)
    path.reverse()
    
    return path, distances[end]

# --- 3. Funciones de Ayuda ---

def format_node(node):
    """Convierte una tupla (54, 14) al formato 'C54, K14'"""
    if not isinstance(node, tuple) or len(node) != 2:
        return str(node)
    return f"C{node[0]}, K{node[1]}"

# --- 4. Clase Principal de la Aplicación (GUI) ---

class App(customtkinter.CTk):
    
    def __init__(self):
        super().__init__()
        self.title("Calculador de Rutas (Cargado desde JSON)")
        self.geometry("1000x850")
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- 1. CREAR LA GUI PRIMERO ---
        self.button_frame = customtkinter.CTkFrame(self)
        self.button_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.graph_frame = customtkinter.CTkFrame(self)
        self.graph_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.results_frame = customtkinter.CTkFrame(self)
        self.results_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.label_destino = customtkinter.CTkLabel(self.button_frame, text="Selecciona un destino:", font=("Arial", 16))
        self.label_destino.pack(side="left", padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(facecolor="#2B2B2B")
        self.fig.set_size_inches(7, 7)
        self.ax.set_facecolor("#2B2B2B")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        # Crear la caja de texto (¡AHORA SÍ EXISTE!)
        self.results_textbox = customtkinter.CTkTextbox(self.results_frame, font=("Courier New", 14), wrap="word")
        self.results_textbox.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # --- 2. INTENTAR CARGAR LA CONFIGURACIÓN ---
        try:
            self.load_config_and_build_graphs("mapa.json")
        except FileNotFoundError:
            print("ERROR: No se encontró el archivo 'mapa.json'")
            self.results_textbox.insert("0.0", "ERROR: No se encontró el archivo 'mapa.json'\n\nAsegúrate de que el archivo está en la misma carpeta.")
            self.results_textbox.configure(state="disabled")
            return
        except Exception as e:
            print(f"Error al cargar 'mapa.json': {e}")
            self.results_textbox.insert("0.0", f"Error al cargar 'mapa.json':\n{e}")
            self.results_textbox.configure(state="disabled")
            return

        # --- 3. AÑADIR LOS BOTONES (ahora que sabemos los destinos) ---
        for destino_nombre in self.DESTINOS.keys():
            button = customtkinter.CTkButton(
                self.button_frame,
                text=destino_nombre,
                command=lambda d=destino_nombre: self.on_button_click(d)
            )
            button.pack(side="left", padx=10, pady=10)

        # --- 4. DIBUJAR EL GRAFO Y MOSTRAR MENSAJE ---
        self.draw_graph()
        self.results_textbox.insert("0.0", "Bienvenido.\n\nMapa cargado desde 'mapa.json'.\n\nEste calculador usa el algoritmo 'Doble Dijkstra'.\n\nSelecciona un destino.")
        self.results_textbox.configure(state="disabled")
    
    # --- 6. FUNCIÓN PARA CARGAR Y CONSTRUIR ---
    def load_config_and_build_graphs(self, filename):
        """
        Lee el archivo JSON y construye los grafos G y DG.
        """
        
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Cargar locaciones y convertirlas a tuplas (necesario para NetworkX)
        self.CASA_JAVIER = tuple(config['casa_javier'])
        self.CASA_ANDREINA = tuple(config['casa_andreina'])
        self.DESTINOS = {name: tuple(coords) for name, coords in config['destinos'].items()}
        
        # Cargar rangos
        self.CALLES = range(config['rango_calles'][0], config['rango_calles'][1])
        self.CARRERAS = range(config['rango_carreras'][0], config['rango_carreras'][1])
        
        # Cargar costos
        COSTO_NORMAL = config['costos']['normal']
        COSTO_CARRERAS_LENTAS = config['costos']['carrera_lenta']
        COSTO_CALLE_LENTA = config['costos']['calle_lenta']
        
        # Cargar reglas (usar sets para búsquedas rápidas)
        carreras_lentas = set(config['reglas_costos']['carreras_lentas'])
        calles_lentas = set(config['reglas_costos']['calles_lentas'])

        # --- Construir los grafos (lógica movida aquí) ---
        self.G = nx.Graph()
        self.pos = {}
        self.node_colors = {}
        edge_colors = {}
        
        for c in self.CALLES:
            for k in self.CARRERAS:
                node = (c, k)
                self.pos[node] = (k, c) 
                self.G.add_node(node)
                self.node_colors[node] = "#CCCCCC"

        # Colorear nodos especiales
        self.node_colors[self.CASA_JAVIER] = "#00FF00"
        self.node_colors[self.CASA_ANDREINA] = "#FF00FF"
        for dest in self.DESTINOS.values():
            self.node_colors[dest] = "#FFFF00"
                
        # Añadir aristas y pesos basados en las reglas del JSON
        for c in self.CALLES:
            for k in self.CARRERAS:
                node = (c, k)
                
                # Vecino Norte/Sur
                if c - 1 in self.CALLES:
                    costo = COSTO_CARRERAS_LENTAS if k in carreras_lentas else COSTO_NORMAL
                    self.G.add_edge(node, (c - 1, k), weight=costo)
                    # --- LÍNEA CORREGIDA ---
                    edge_colors[(node, (c - 1, k))] = "#AAAAAA" # Siempre gris

                # Vecino Este/Oeste
                if k - 1 in self.CARRERAS:
                    costo = COSTO_CALLE_LENTA if c in calles_lentas else COSTO_NORMAL
                    self.G.add_edge(node, (c, k - 1), weight=costo)
                    # --- LÍNEA CORREGIDA ---
                    edge_colors[(node, (c, k - 1))] = "#AAAAAA" # Siempre gris
                    
        self.node_color_list = [self.node_colors[node] for node in self.G.nodes()]
        self.edge_color_list = []
        for u, v in self.G.edges():
            color = edge_colors.get((u, v), edge_colors.get((v, u), "#AAAAAA"))
            self.edge_color_list.append(color)

        # Construir el grafo NO dirigido para el algoritmo
        self.DG = self.G.to_directed()
    
    # --- 7. OTRAS FUNCIONES ---
    def draw_graph(self, javier_path=None, andreina_path=None):
        self.ax.clear()
        # Usar self.CARRERAS y self.CALLES
        for k in self.CARRERAS: self.ax.text(k, 49.7, f"K{k}", color="white", ha="center", va="top", fontsize=10)
        for c in self.CALLES: self.ax.text(9.8, c, f"C{c}", color="white", ha="right", va="center", fontsize=10)
        
        nx.draw_networkx(
            self.G, self.pos, ax=self.ax,
            node_color=self.node_color_list, # Usar la lista pre-calculada
            edge_color=self.edge_color_list, # Usar la lista pre-calculada
            with_labels=False, node_size=250
        )
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, ax=self.ax,
            font_color='white', font_size=8,
            bbox=dict(facecolor=self.ax.get_facecolor(), edgecolor='none', pad=0.2, alpha=0.8)
        )
        
        # Usar self.CASA_JAVIER, self.CASA_ANDREINA, self.DESTINOS
        labels = {
            self.CASA_JAVIER: "Javier", 
            self.CASA_ANDREINA: "Andreína",
            **{dest: name for name, dest in self.DESTINOS.items()}
        }
        nx.draw_networkx_labels(
            self.G, self.pos, labels=labels, ax=self.ax, 
            font_color='white', font_size=10
        )
        if javier_path:
            path_edges = list(zip(javier_path, javier_path[1:]))
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax, edgelist=path_edges,
                edge_color="#00FFFF", width=3.0
            )
        if andreina_path:
            path_edges = list(zip(andreina_path, andreina_path[1:]))
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax, edgelist=path_edges,
                edge_color="#FF00FF", width=3.0
            )
        self.ax.set_axis_off()
        self.canvas.draw()


    def get_path_cost_from_original_graph(self, path):
        # Esta función ya usaba self.G, así que está bien
        cost = 0
        for u, v in zip(path, path[1:]):
            if self.G.has_edge(u, v):
                cost += self.G[u][v]['weight']
            else:
                return float('inf') 
        return cost


    def on_button_click(self, destino_nombre):
        # Usar self.DESTINOS, self.CASA_JAVIER, self.CASA_ANDREINA
        destino_coords = self.DESTINOS[destino_nombre]
        
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("0.0", "end")
        output = f"--- DESTINO: {destino_nombre} ---\n"
        output += "[Calculando con 'Heurística Doble Dijkstra' (No-Dirigido)...]\n\n"
        self.results_textbox.insert("0.0", output)

        # --- ESCENARIO 1: JAVIER ELIGE PRIMERO ---
        output += "--- Escenario 1: Javier elige primero ---\n"
        ruta_j1, tiempo_j1 = find_shortest_path_dijkstra(self.DG, self.CASA_JAVIER, destino_coords)
        
        if tiempo_j1 == float('inf'):
            output += "ERROR: Javier no puede llegar al destino.\n"
            self.results_textbox.insert("0.0", output)
            self.results_textbox.configure(state="disabled")
            return

        G_temp1 = self.DG.copy()
        for u, v in zip(ruta_j1, ruta_j1[1:]):
            if G_temp1.has_edge(u, v): G_temp1.remove_edge(u, v)
            if G_temp1.has_edge(v, u): G_temp1.remove_edge(v, u)
        
        ruta_a1, tiempo_a1 = find_shortest_path_dijkstra(G_temp1, self.CASA_ANDREINA, destino_coords)
        
        total_tiempo1 = tiempo_j1 + tiempo_a1
        output += f"  Ruta Javier: {tiempo_j1} min\n"
        output += f"  Ruta Andreína: {tiempo_a1} min\n"
        output += f"  Tiempo Total Escenario 1: {total_tiempo1} min\n\n"

        # --- ESCENARIO 2: ANDREÍNA ELIGE PRIMERO ---
        output += "--- Escenario 2: Andreína elige primero ---\n"
        ruta_a2, tiempo_a2 = find_shortest_path_dijkstra(self.DG, self.CASA_ANDREINA, destino_coords)

        if tiempo_a2 == float('inf'):
            output += "ERROR: Andreína no puede llegar al destino.\n"
            self.results_textbox.insert("0.0", output)
            self.results_textbox.configure(state="disabled")
            return

        G_temp2 = self.DG.copy()
        for u, v in zip(ruta_a2, ruta_a2[1:]):
            if G_temp2.has_edge(u, v): G_temp2.remove_edge(u, v)
            if G_temp2.has_edge(v, u): G_temp2.remove_edge(v, u)
        
        ruta_j2, tiempo_j2 = find_shortest_path_dijkstra(G_temp2, self.CASA_JAVIER, destino_coords)
        
        total_tiempo2 = tiempo_a2 + tiempo_j2
        output += f"  Ruta Andreína: {tiempo_a2} min\n"
        output += f"  Ruta Javier: {tiempo_j2} min\n"
        output += f"  Tiempo Total Escenario 2: {total_tiempo2} min\n\n"
        
        # --- DECISIÓN FINAL ---
        output += "--- Decisión Final --- \n"
        
        if total_tiempo1 == float('inf') and total_tiempo2 == float('inf'):
            output += "ERROR CRÍTICO: No hay solución. Es imposible que ambos lleguen sin compartir una calle."
            self.draw_graph()
            self.results_textbox.delete("0.0", "end")
            self.results_textbox.insert("0.0", output)
            self.results_textbox.configure(state="disabled")
            return

        if total_tiempo1 <= total_tiempo2:
            output += f"Se elige el Escenario 1 (Total: {total_tiempo1} min) por ser el más rápido.\n\n"
            ruta_j_final, tiempo_j_final = ruta_j1, tiempo_j1
            ruta_a_final, tiempo_a_final = ruta_a1, tiempo_a1
        else:
            output += f"Se elige el Escenario 2 (Total: {total_tiempo2} min) por ser el más rápido.\n\n"
            ruta_j_final, tiempo_j_final = ruta_j2, tiempo_j2
            ruta_a_final, tiempo_a_final = ruta_a2, tiempo_a2

        tiempo_j_final = self.get_path_cost_from_original_graph(ruta_j_final)
        tiempo_a_final = self.get_path_cost_from_original_graph(ruta_a_final)

        ruta_j_str = " -> ".join([format_node(n) for n in ruta_j_final])
        ruta_a_str = " -> ".join([format_node(n) for n in ruta_a_final])

        # Usar self.CASA_JAVIER y self.CASA_ANDREINA para los formatos
        output += f"🧍‍♂️ JAVIER (desde {format_node(self.CASA_JAVIER)}):\n"
        output += f"  Ruta: {ruta_j_str}\n"
        output += f"  Tiempo: {tiempo_j_final} minutos\n\n"

        output += f"🧍‍♀️ ANDREÍNA (desde {format_node(self.CASA_ANDREINA)}):\n"
        output += f"  Ruta: {ruta_a_str}\n"
        output += f"  Tiempo: {tiempo_a_final} minutos\n\n"
        
        output += f"TIEMPO TOTAL (J+A): {tiempo_j_final + tiempo_a_final} minutos\n\n"
        
        output += "⏱️ SINCRONIZACIÓN:\n"
        if tiempo_j_final > tiempo_a_final:
            diferencia = tiempo_j_final - tiempo_a_final
            output += f"  Andreína debe salir {diferencia} minutos antes que Javier.\n"
        elif tiempo_a_final > tiempo_j_final:
            diferencia = tiempo_a_final - tiempo_j_final
            output += f"  Javier debe salir {diferencia} minutos antes que Andreína.\n"
        else:
            output += "  Ambos pueden salir al mismo tiempo.\n"
            
        self.draw_graph(javier_path=ruta_j_final, andreina_path=ruta_a_final)
        
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", output)
        self.results_textbox.configure(state="disabled")

# --- 8. Punto de Entrada Principal (Sin cambios) ---
if __name__ == "__main__":
    app = App()
    app.mainloop()