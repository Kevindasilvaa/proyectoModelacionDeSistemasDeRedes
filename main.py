import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter
import heapq
import json

# --- Configuración Inicial de la Apariencia ---
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

# --- Implementación Manual de Dijkstra ---
def find_shortest_path_dijkstra(graph, start, end):
    """
    ¿Cómo funciona?
    1. Empezamos desde el nodo inicial con distancia 0
    2. Exploramos todos los vecinos y calculamos la distancia acumulada
    3. Siempre elegimos el nodo con menor distancia acumulada (usando una cola de prioridad)
    4. Actualizamos las distancias si encontramos un camino más corto
    5. Guardamos el nodo "anterior" para reconstruir la ruta al final
    
    Parámetros:
        graph: El grafo de NetworkX con nodos y aristas (es una lista de adyacencia)
        start: Nodo de inicio (tupla con coordenadas)
        end: Nodo de destino (tupla con coordenadas)
    
    Retorna:
        path: Lista de nodos que forman la ruta [inicio, ..., fin]
        distance: Costo total de la ruta
    """
    
    # PASO 1: Inicialización de estructuras de datos
    # ============================================
    
    # 'distances': Diccionario que guarda la distancia MÁS CORTA conocida desde 'start' hasta cada nodo
    # Al inicio, todos los nodos tienen distancia infinita (no sabemos cómo llegar)
    distances = {node: float('inf') for node in graph.nodes()}
    
    # 'predecessors': Diccionario para reconstruir el camino
    # Guarda "¿desde qué nodo llegué aquí en la ruta más corta?"
    # Ejemplo: si predecessors[B] = A, significa que llegamos a B desde A
    predecessors = {node: None for node in graph.nodes()}
    
    # Validación: Si el nodo de inicio no existe, no hay ruta posible
    if start not in distances:
        return [], float('inf')
    
    # La distancia desde el inicio hasta sí mismo es 0
    distances[start] = 0
    
    # PASO 2: Cola de Prioridad (min-heap)
    # ====================================
    
    # La cola almacena tuplas: (distancia_acumulada, nodo)
    # heapq siempre nos dará el nodo con la MENOR distancia acumulada
    # Esto es clave: siempre procesamos el nodo "más cercano" primero
    priority_queue = [(0, start)]  # Empezamos con el nodo inicial (distancia 0)
    
    # PASO 3: Bucle Principal - Exploración de nodos
    # ==============================================
    
    while priority_queue:
        # Extraemos el nodo con menor distancia de la cola
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # OPTIMIZACIÓN: Si ya encontramos un camino más corto a este nodo,
        # ignoramos esta entrada antigua de la cola
        # Esto puede pasar porque no borramos entradas viejas de la cola
        if current_distance > distances[current_node]:
            continue
        
        # OPTIMIZACIÓN: Si llegamos al destino, ya encontramos la ruta más corta
        # No necesitamos seguir explorando (early stopping)
        if current_node == end:
            break

        # PASO 4: Relajación de Aristas (explorar vecinos)
        # ================================================
        
        # Revisamos todos los vecinos del nodo actual
        for neighbor in graph.neighbors(current_node):
            # Verificación de seguridad (normalmente no es necesaria)
            if not graph.has_edge(current_node, neighbor):
                continue
            
            # Obtenemos el COSTO (peso) de ir del nodo actual al vecino
            # Este es el tiempo que toma caminar por esa cuadra
            weight = graph[current_node][neighbor]['weight']
            
            # Calculamos la distancia total: distancia hasta aquí + peso de esta arista
            distance = current_distance + weight
            
            # RELAJACIÓN: Si este nuevo camino es MÁS CORTO que el mejor que teníamos...
            if distance < distances[neighbor]:
                # ...actualizamos la distancia más corta conocida
                distances[neighbor] = distance
                
                # ...guardamos que llegamos al vecino desde current_node
                predecessors[neighbor] = current_node
                
                # ...agregamos el vecino a la cola para explorarlo después
                # (con su nueva distancia más corta)
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # PASO 5: Reconstrucción de la Ruta
    # ==================================
    
    # Si la distancia al destino sigue siendo infinito, significa que NO HAY RUTA
    if distances[end] == float('inf'):
        return [], float('inf')
    
    # Reconstruimos el camino yendo HACIA ATRÁS desde el destino hasta el inicio
    path = []
    current = end
    
    # Vamos retrocediendo usando el diccionario 'predecessors'
    while current != start:
        # Si current es None, algo salió mal (no debería pasar si distances[end] != inf)
        if current is None:
            return [], float('inf')
        
        path.append(current)  # Agregamos el nodo actual al camino
        current = predecessors.get(current)  # Retrocedemos al nodo anterior
    
    path.append(start)  # No olvidamos agregar el nodo de inicio
    path.reverse()  # Invertimos para tener [inicio -> ... -> fin] en lugar de [fin -> ... -> inicio]
    
    # Retornamos la ruta completa y su costo total
    return path, distances[end]

# --- Funciones de Ayuda ---
def format_node(node):
    if not isinstance(node, tuple) or len(node) != 2:
        return str(node)
    return f"C{node[0]}, K{node[1]}"

# --- Clase Principal de la Aplicación ---
class App(customtkinter.CTk):
    
    def __init__(self):
        super().__init__()
        
        # --- Configuración de la Ventana ---
        self.title("🗺️ Calculador de Rutas Óptimas")
        self.geometry("1400x1000")  # Tamaño por defecto antes de maximizar
        
        # --- Colores personalizados para mantener consistencia visual ---
        self.bg_color = "#1a1a1a"        # Color de fondo principal (gris muy oscuro)
        self.frame_color = "#252525"     # Color de los paneles/frames (gris oscuro)
        self.accent_color = "#3b82f6"    # Color de acento azul para botones y bordes
        self.success_color = "#10b981"   # Color verde para Javier
        self.warning_color = "#f59e0b"   # Color naranja/amarillo para destinos
        
        # --- Grid Configuration (Sistema de rejilla para organizar elementos) ---
        # Las filas con weight=0 no se expanden, las de weight=1 sí se expanden
        self.grid_rowconfigure(0, weight=0)  # Fila 0: Header (título) - tamaño fijo
        self.grid_rowconfigure(1, weight=0)  # Fila 1: Botones de destino - tamaño fijo
        self.grid_rowconfigure(2, weight=1)  # Fila 2: Mapa y resultados - se expande para llenar espacio
        
        # Las columnas con weight=1 se expanden proporcionalmente
        self.grid_columnconfigure(0, weight=1)  # Columna 0: Mapa (izquierda)
        self.grid_columnconfigure(1, weight=1)  # Columna 1: Resultados (derecha)

        # --- HEADER (Encabezado superior con título principal) ---
        self.header_frame = customtkinter.CTkFrame(self, fg_color=self.frame_color, corner_radius=15)
        # grid() posiciona el elemento: row=fila, column=columna, columnspan=cuántas columnas ocupa
        # sticky="ew" significa que se expande Este-Oeste (horizontalmente)
        # padx/pady son los márgenes externos
        self.header_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        
        self.title_label = customtkinter.CTkLabel(
            self.header_frame, 
            text="🗺️ CALCULADOR DE RUTAS ÓPTIMAS",
            font=("Arial Black", 28, "bold"),
            text_color="#ffffff"
        )
        self.title_label.pack(pady=20)
        
        self.subtitle_label = customtkinter.CTkLabel(
            self.header_frame,
            text="Sistema de navegación con algoritmo Doble Dijkstra",
            font=("Arial", 14),
            text_color="#a0a0a0"
        )
        self.subtitle_label.pack(pady=(0, 15))

        # --- PANEL DE CONTROL (Botones de destino) ---
        self.control_frame = customtkinter.CTkFrame(self, fg_color=self.frame_color, corner_radius=15)
        self.control_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        self.label_destino = customtkinter.CTkLabel(
            self.control_frame,
            text="📍 Selecciona un Destino:",
            font=("Arial", 18, "bold"),
            text_color=self.accent_color
        )
        self.label_destino.pack(side="left", padx=20, pady=15)
        
        # --- MAPA (Panel izquierdo que muestra el grafo) ---
        self.graph_frame = customtkinter.CTkFrame(self, fg_color=self.frame_color, corner_radius=15)
        # sticky="nsew" hace que se expanda en todas direcciones (Norte, Sur, Este, Oeste)
        self.graph_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="nsew")
        
        # Título del mapa
        self.map_title = customtkinter.CTkLabel(
            self.graph_frame,
            text="🗺️ MAPA DE RUTAS",
            font=("Arial", 16, "bold"),
            text_color="#ffffff"
        )
        self.map_title.pack(pady=(15, 5))
        
        # --- RESULTADOS (Panel derecho que muestra el análisis) ---
        self.results_frame = customtkinter.CTkFrame(self, fg_color=self.frame_color, corner_radius=15)
        self.results_frame.grid(row=2, column=1, padx=(0, 20), pady=(10, 20), sticky="nsew")
        
        # Título de resultados
        self.results_title = customtkinter.CTkLabel(
            self.results_frame,
            text="📊 ANÁLISIS DE RUTAS",
            font=("Arial", 16, "bold"),
            text_color="#ffffff"
        )
        self.results_title.pack(pady=(15, 10))
        
        # --- Matplotlib Setup (Configuración del gráfico) ---
        # Creamos una figura (fig) y ejes (ax) para dibujar el grafo
        self.fig, self.ax = plt.subplots(facecolor=self.frame_color)
        self.fig.set_size_inches(9, 8)  # Tamaño del gráfico en pulgadas
        self.ax.set_facecolor("#1e1e1e")  # Color de fondo del área de dibujo
        
        # FigureCanvasTkAgg es el "puente" que permite mostrar matplotlib dentro de CustomTkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        # pack() organiza el widget para que llene todo el espacio disponible
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        # Caja de texto con mejor estilo
        self.results_textbox = customtkinter.CTkTextbox(
            self.results_frame,
            font=("Consolas", 13),
            wrap="word",
            fg_color="#1e1e1e",
            border_width=2,
            border_color=self.accent_color
        )
        self.results_textbox.pack(side="top", fill="both", expand=True, padx=15, pady=(0, 15))

        # --- CARGAR CONFIGURACIÓN ---
        try:
            self.load_config_and_build_graphs("mapa.json")
        except FileNotFoundError:
            self.show_error("ERROR: No se encontró el archivo 'mapa.json'\n\n⚠️ Asegúrate de que el archivo está en la misma carpeta.")
            return
        except Exception as e:
            self.show_error(f"Error al cargar 'mapa.json':\n{e}")
            return

        # --- CREAR BOTONES DE DESTINO (uno por cada destino en el JSON) ---
        self.buttons = []
        for i, destino_nombre in enumerate(self.DESTINOS.keys()):
            button = customtkinter.CTkButton(
                self.control_frame,
                text=f"📍 {destino_nombre}",
                font=("Arial", 14, "bold"),
                height=45,
                corner_radius=10,
                fg_color=self.accent_color,        # Color normal del botón
                hover_color="#2563eb",             # Color cuando pasas el mouse por encima
                # lambda con d=destino_nombre "captura" el valor actual en cada iteración
                # Sin esto, todos los botones llamarían con el último destino
                command=lambda d=destino_nombre: self.on_button_click(d)
            )
            button.pack(side="left", padx=8, pady=15)  # side="left" los alinea horizontalmente
            self.buttons.append(button)

        # --- MENSAJE INICIAL ---
        self.draw_graph()
        welcome_msg = """╔════════════════════════════════════════╗
║     🎉 BIENVENIDO AL CALCULADOR      ║
╚════════════════════════════════════════╝

✅ Mapa cargado exitosamente desde 'mapa.json'

🧮 Este sistema utiliza el algoritmo 
   'Doble Dijkstra' para encontrar las 
   rutas más eficientes.

📍 Selecciona un destino para comenzar
   el análisis de rutas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ℹ️ El sistema calculará automáticamente:
   • Ruta óptima para Javier
   • Ruta óptima para Andreína  
   • Tiempo total minimizado
   • Sincronización de salida
"""
        self.results_textbox.insert("0.0", welcome_msg)
        self.results_textbox.configure(state="disabled")
        
        # Maximizar la ventana DESPUÉS de que todo esté construido
        # Usamos after() para ejecutarlo en el siguiente ciclo del event loop
        self.after(100, lambda: self.state('zoomed'))
    
    def show_error(self, message):
        self.results_textbox.insert("0.0", f"❌ {message}")
        self.results_textbox.configure(state="disabled")
    
    def load_config_and_build_graphs(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.CASA_JAVIER = tuple(config['casa_javier'])
        self.CASA_ANDREINA = tuple(config['casa_andreina'])
        self.DESTINOS = {name: tuple(coords) for name, coords in config['destinos'].items()}
        
        self.CALLES = range(config['rango_calles'][0], config['rango_calles'][1])
        self.CARRERAS = range(config['rango_carreras'][0], config['rango_carreras'][1])
        
        COSTO_NORMAL = config['costos']['normal']
        COSTO_CARRERAS_LENTAS = config['costos']['carrera_lenta']
        COSTO_CALLE_LENTA = config['costos']['calle_lenta']
        
        carreras_lentas = set(config['reglas_costos']['carreras_lentas'])
        calles_lentas = set(config['reglas_costos']['calles_lentas'])

        self.G = nx.Graph()
        self.pos = {}
        self.node_colors = {}
        edge_colors = {}
        
        for c in self.CALLES:
            for k in self.CARRERAS:
                node = (c, k)
                self.pos[node] = (k, c)
                self.G.add_node(node)
                self.node_colors[node] = "#4a5568"

        self.node_colors[self.CASA_JAVIER] = "#10b981"
        self.node_colors[self.CASA_ANDREINA] = "#ec4899"
        for dest in self.DESTINOS.values():
            self.node_colors[dest] = "#f59e0b"
                
        for c in self.CALLES:
            for k in self.CARRERAS:
                node = (c, k)
                
                if c - 1 in self.CALLES:
                    costo = COSTO_CARRERAS_LENTAS if k in carreras_lentas else COSTO_NORMAL
                    self.G.add_edge(node, (c - 1, k), weight=costo)
                    edge_colors[(node, (c - 1, k))] = "#3a3a3a"

                if k - 1 in self.CARRERAS:
                    costo = COSTO_CALLE_LENTA if c in calles_lentas else COSTO_NORMAL
                    self.G.add_edge(node, (c, k - 1), weight=costo)
                    edge_colors[(node, (c, k - 1))] = "#3a3a3a"
                        
        self.node_color_list = [self.node_colors[node] for node in self.G.nodes()]
        
        self.edge_color_list = []
        for u, v in self.G.edges():
            color = edge_colors.get((u, v), edge_colors.get((v, u), "#3a3a3a"))
            self.edge_color_list.append(color)
    
    def draw_graph(self, javier_path=None, andreina_path=None):
        self.ax.clear()
        
        # Etiquetas mejoradas
        for k in self.CARRERAS:
            self.ax.text(k, 49.7, f"K{k}", color="#60a5fa", ha="center", va="top", 
                        fontsize=11, fontweight="bold")
        for c in self.CALLES:
            self.ax.text(9.8, c, f"C{c}", color="#60a5fa", ha="right", va="center", 
                        fontsize=11, fontweight="bold")
        
        nx.draw_networkx(
            self.G, self.pos, ax=self.ax,
            node_color=self.node_color_list,
            edge_color=self.edge_color_list,
            with_labels=False,
            node_size=300,
            width=2
        )
        
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, ax=self.ax,
            font_color='#94a3b8', font_size=9, font_weight="bold",
            bbox=dict(facecolor='#1e1e1e', edgecolor='none', pad=0.3, alpha=0.9)
        )
        
        labels = {
            self.CASA_JAVIER: "Javier",
            self.CASA_ANDREINA: "Andreína",
            **{dest: name for name, dest in self.DESTINOS.items()}
        }
        nx.draw_networkx_labels(
            self.G, self.pos, labels=labels, ax=self.ax,
            font_color='white', font_size=11, font_weight="bold"
        )
        
        if javier_path:
            path_edges = list(zip(javier_path, javier_path[1:]))
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax, edgelist=path_edges,
                edge_color="#10b981", width=5.0, style="solid"
            )
        if andreina_path:
            path_edges = list(zip(andreina_path, andreina_path[1:]))
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax, edgelist=path_edges,
                edge_color="#ec4899", width=5.0, style="solid"
            )
            
        self.ax.set_axis_off()
        self.canvas.draw()

    def get_path_cost_from_original_graph(self, path):
        cost = 0
        for u, v in zip(path, path[1:]):
            if self.G.has_edge(u, v):
                cost += self.G[u][v]['weight']
            else:
                return float('inf')
        return cost

    def on_button_click(self, destino_nombre):
        destino_coords = self.DESTINOS[destino_nombre]
        
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("0.0", "end")
        
        output = f"""╔════════════════════════════════════════╗
║     📍 DESTINO: {destino_nombre.upper()}
╚════════════════════════════════════════╝

🔄 Calculando rutas óptimas...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        self.results_textbox.insert("0.0", output)
        self.results_textbox.update()

        # --- ESCENARIO 1 ---
        output += "┌─ ESCENARIO 1: Javier elige primero ─┐\n"
        
        ruta_j1, tiempo_j1 = find_shortest_path_dijkstra(self.G, self.CASA_JAVIER, destino_coords)
        
        if tiempo_j1 == float('inf'):
            output += "❌ ERROR: Javier no puede llegar\n"
            self.results_textbox.delete("0.0", "end")
            self.results_textbox.insert("0.0", output)
            self.results_textbox.configure(state="disabled")
            return

        G_temp1 = self.G.copy()
        for u, v in zip(ruta_j1, ruta_j1[1:]):
            if G_temp1.has_edge(u, v):
                G_temp1.remove_edge(u, v)
        
        ruta_a1, tiempo_a1 = find_shortest_path_dijkstra(G_temp1, self.CASA_ANDREINA, destino_coords)
        
        total_tiempo1 = tiempo_j1 + tiempo_a1
        output += f"  🧍‍♂️ Javier:   {tiempo_j1} min\n"
        output += f"  🧍‍♀️ Andreína: {tiempo_a1} min\n"
        output += f"  ⏱️  Total:    {total_tiempo1} min\n"
        output += "└────────────────────────────────────┘\n\n"

        # --- ESCENARIO 2 ---
        output += "┌─ ESCENARIO 2: Andreína elige primero ┐\n"
        
        ruta_a2, tiempo_a2 = find_shortest_path_dijkstra(self.G, self.CASA_ANDREINA, destino_coords)

        if tiempo_a2 == float('inf'):
            output += "❌ ERROR: Andreína no puede llegar\n"
            self.results_textbox.delete("0.0", "end")
            self.results_textbox.insert("0.0", output)
            self.results_textbox.configure(state="disabled")
            return

        G_temp2 = self.G.copy()
        for u, v in zip(ruta_a2, ruta_a2[1:]):
            if G_temp2.has_edge(u, v):
                G_temp2.remove_edge(u, v)
        
        ruta_j2, tiempo_j2 = find_shortest_path_dijkstra(G_temp2, self.CASA_JAVIER, destino_coords)
        
        total_tiempo2 = tiempo_a2 + tiempo_j2
        output += f"  🧍‍♀️ Andreína: {tiempo_a2} min\n"
        output += f"  🧍‍♂️ Javier:   {tiempo_j2} min\n"
        output += f"  ⏱️  Total:    {total_tiempo2} min\n"
        output += "└────────────────────────────────────┘\n\n"
        
        # --- DECISIÓN ---
        output += "╔════════════════════════════════════════╗\n"
        output += "║        ✨ SOLUCIÓN ÓPTIMA ✨          ║\n"
        output += "╚════════════════════════════════════════╝\n\n"
        
        if total_tiempo1 == float('inf') and total_tiempo2 == float('inf'):
            output += "❌ ERROR: No hay solución posible\n"
            self.draw_graph()
            self.results_textbox.delete("0.0", "end")
            self.results_textbox.insert("0.0", output)
            self.results_textbox.configure(state="disabled")
            return

        if total_tiempo1 <= total_tiempo2:
            output += f"✅ Escenario 1 seleccionado\n"
            output += f"   Total: {total_tiempo1} min\n\n"
            ruta_j_final, tiempo_j_final = ruta_j1, tiempo_j1
            ruta_a_final, tiempo_a_final = ruta_a1, tiempo_a1
        else:
            output += f"✅ Escenario 2 seleccionado\n"
            output += f"   Total: {total_tiempo2} min\n\n"
            ruta_j_final, tiempo_j_final = ruta_j2, tiempo_j2
            ruta_a_final, tiempo_a_final = ruta_a2, tiempo_a2

        tiempo_j_final = self.get_path_cost_from_original_graph(ruta_j_final)
        tiempo_a_final = self.get_path_cost_from_original_graph(ruta_a_final)

        ruta_j_str = " → ".join([format_node(n) for n in ruta_j_final])
        ruta_a_str = " → ".join([format_node(n) for n in ruta_a_final])

        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        output += f"🧍‍♂️ JAVIER ({format_node(self.CASA_JAVIER)})\n"
        output += f"   Ruta: {ruta_j_str}\n"
        output += f"   ⏱️  {tiempo_j_final} minutos\n\n"

        output += f"🧍‍♀️ ANDREÍNA ({format_node(self.CASA_ANDREINA)})\n"
        output += f"   Ruta: {ruta_a_str}\n"
        output += f"   ⏱️  {tiempo_a_final} minutos\n\n"
        
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"⏱️  TIEMPO TOTAL: {tiempo_j_final + tiempo_a_final} min\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        output += "⏰ SINCRONIZACIÓN DE SALIDA:\n\n"
        if tiempo_j_final > tiempo_a_final:
            diferencia = tiempo_j_final - tiempo_a_final
            output += f"  🏁 Javier sale PRIMERO\n"
            output += f"  ⏳ Andreína sale {diferencia} min DESPUÉS\n"
        elif tiempo_a_final > tiempo_j_final:
            diferencia = tiempo_a_final - tiempo_j_final
            output += f"  🏁 Andreína sale PRIMERO\n"
            output += f"  ⏳ Javier sale {diferencia} min DESPUÉS\n"
        else:
            output += "  🏁 Ambos salen AL MISMO TIEMPO\n"
            
        self.draw_graph(javier_path=ruta_j_final, andreina_path=ruta_a_final)
        
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", output)
        self.results_textbox.configure(state="disabled")

# --- Punto de Entrada ---
if __name__ == "__main__":
    app = App()
    app.mainloop()