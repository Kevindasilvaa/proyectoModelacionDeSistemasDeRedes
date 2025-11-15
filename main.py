import networkx as nx # La librería principal para crear, manipular y estudiar grafos.
import matplotlib.pyplot as plt # La librería principal para crear gráficos y visualizaciones.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Un "puente" para incrustar gráficos de Matplotlib en una ventana de Tkinter.
import customtkinter # La librería que crea la interfaz gráfica de usuario (GUI) moderna.
import heapq # Librería para implementar colas de prioridad (min-heap), esencial para el algoritmo de Dijkstra.
import json # Librería para leer y escribir archivos en formato JSON (JavaScript Object Notation).

# --- Configuración Inicial de la Apariencia de la GUI ---
customtkinter.set_appearance_mode("Dark") # Establece el modo oscuro para la ventana.
customtkinter.set_default_color_theme("blue") # Establece el color de los botones en azul.

# --- 2. Implementación Manual de Dijkstra ---
def find_shortest_path_dijkstra(graph, start, end):
    """
    Implementación manual del algoritmo de Dijkstra.
    Busca la ruta más corta (con menor peso/costo) desde un nodo 'start' a un nodo 'end'
    en un grafo dirigido ('graph').
    Devuelve la ruta (lista de nodos) y el costo total.
    """
    
    # 1. Inicialización:
    # 'distances': Diccionario para almacenar el costo MÍNIMO encontrado hasta ahora para CADA nodo.
    # Se inicializan todos en infinito, excepto el de inicio.
    distances = {node: float('inf') for node in graph.nodes()}
    
    # 'predecessors': Diccionario para reconstruir la ruta.
    # Almacena el nodo "anterior" en la ruta más corta. Ej: {B: A, C: B}
    predecessors = {node: None for node in graph.nodes()}
    
    # Si el nodo de inicio no existe en el grafo, no hay ruta.
    if start not in distances:
        return [], float('inf')
        
    # El costo para llegar al nodo de inicio desde sí mismo es 0.
    distances[start] = 0
    
    # 2. Cola de Prioridad (min-heap):
    # Almacena tuplas de (costo_acumulado_para_llegar_a_un_nodo, el_nodo).
    # 'heapq' se asegura de que siempre podamos sacar (heappop) el nodo con el menor costo acumulado.
    priority_queue = [(0, start)]
    
    # 3. Bucle Principal de Dijkstra:
    # Mientras haya nodos por explorar en la cola...
    while priority_queue:
        # Sacamos el nodo con el menor costo actual de la cola.
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Optimización: Si el costo que sacamos de la cola es MAYOR
        # que uno que ya teníamos guardado, significa que encontramos
        # una ruta más rápida a este nodo en una iteración anterior. Lo ignoramos.
        if current_distance > distances[current_node]:
            continue
            
        # Optimización: Si el nodo actual es el destino, hemos encontrado
        # la ruta más corta hacia él. Podemos detenernos.
        if current_node == end:
            break

        # 4. Relajación de Aristas (Explorar vecinos):
        # Iteramos sobre todos los vecinos del nodo actual.
        for neighbor in graph.neighbors(current_node):
            # (Control de seguridad, puede no ser necesario si el grafo está bien construido)
            if not graph.has_edge(current_node, neighbor):
                continue
                
            # Obtenemos el costo (peso) de ir desde el nodo actual a este vecino.
            weight = graph[current_node][neighbor]['weight']
            # Calculamos el costo total desde el INICIO (start) hasta este VECINO.
            distance = current_distance + weight
            
            # Si esta nueva ruta es MÁS CORTA que la mejor que teníamos...
            if distance < distances[neighbor]:
                # ...actualizamos la distancia más corta para ese vecino.
                distances[neighbor] = distance
                # ...guardamos que llegamos a él a través del 'current_node'.
                predecessors[neighbor] = current_node
                # ...agregamos al vecino a la cola de prioridad para explorarlo después.
                heapq.heappush(priority_queue, (distance, neighbor))
                                    
    # 5. Reconstrucción de la Ruta:
    # Si la distancia al 'end' sigue siendo infinito, significa que no se encontró ruta.
    if distances[end] == float('inf'):
        return [], float('inf')
        
    # Para construir la ruta, empezamos desde el final ('end')
    # y vamos hacia atrás usando el diccionario 'predecessors'.
    path = []
    current = end
    while current != start:
        # Si 'current' es None, algo salió mal (ruta rota).
        if current is None:
            return [], float('inf')
        path.append(current)
        current = predecessors.get(current) # Retrocedemos al nodo anterior.
    
    path.append(start) # Añadimos el nodo de inicio.
    path.reverse() # Invertimos la ruta para que sea [start, ..., end].
    
    # Devolvemos la ruta encontrada y su costo total.
    return path, distances[end]

# --- 3. Funciones de Ayuda ---

def format_node(node):
    """
    Función de ayuda para formatear un nodo.
    Convierte una tupla (ej. (54, 14)) al formato de texto 'C54, K14'.
    """
    # Si no es una tupla de 2 elementos, solo lo convierte a string.
    if not isinstance(node, tuple) or len(node) != 2:
        return str(node)
    # Formato C[Calle], K[Carrera]
    return f"C{node[0]}, K{node[1]}"

# --- 4. Clase Principal de la Aplicación (GUI) ---

# Nuestra aplicación hereda de customtkinter.CTk, convirtiéndose en una ventana.
class App(customtkinter.CTk):
    
    # El constructor de la clase, se llama al crear 'App()'.
    def __init__(self):
        super().__init__() # Llama al constructor de la clase padre (CTk)
        
        # --- Configuración de la Ventana Principal ---
        self.title("Calculador de Rutas (Cargado desde JSON)") # Título de la ventana
        self.geometry("1000x850") # Tamaño inicial (ancho x alto)
        
        # --- Configuración del Layout (Rejilla/Grid) ---
        # Fila 0 (botones) no se expande.
        self.grid_rowconfigure(0, weight=0) 
        # Fila 1 (mapa y resultados) sí se expande para llenar el espacio.
        self.grid_rowconfigure(1, weight=1) 
        # Columna 0 (mapa) se expande.
        self.grid_columnconfigure(0, weight=1)
        # Columna 1 (resultados) se expande.
        self.grid_columnconfigure(1, weight=1)

        # --- 1. CREAR LOS 'WIDGETS' DE LA GUI ---
        
        # 'Frame' (contenedor) para los botones de destino.
        self.button_frame = customtkinter.CTkFrame(self)
        self.button_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # 'Frame' (contenedor) para el gráfico del mapa.
        self.graph_frame = customtkinter.CTkFrame(self)
        self.graph_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # 'Frame' (contenedor) para la caja de texto de resultados.
        self.results_frame = customtkinter.CTkFrame(self)
        self.results_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        # Etiqueta de texto "Selecciona un destino:".
        self.label_destino = customtkinter.CTkLabel(self.button_frame, text="Selecciona un destino:", font=("Arial", 16))
        self.label_destino.pack(side="left", padx=10, pady=10) # 'pack' para organizar dentro del button_frame.
        
        # --- Preparación de Matplotlib para el Grafo ---
        # Creamos la 'figura' (el lienzo) y los 'ejes' (el área de dibujo) de Matplotlib.
        self.fig, self.ax = plt.subplots(facecolor="#2B2B2B") # Fondo oscuro
        self.fig.set_size_inches(7, 7) # Tamaño del lienzo
        self.ax.set_facecolor("#2B2B2B") # Fondo del área de dibujo
        
        # --- Incrustar Matplotlib en CustomTkinter ---
        # Creamos el 'Canvas' de TkAgg, que actúa como puente.
        # Le decimos que 'self.fig' (la figura) debe dibujarse en 'self.graph_frame' (el contenedor).
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True) # Empaquetamos el widget del lienzo.
        
        # Creamos la caja de texto para los resultados.
        self.results_textbox = customtkinter.CTkTextbox(self.results_frame, font=("Courier New", 14), wrap="word")
        self.results_textbox.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # --- 2. INTENTAR CARGAR LA CONFIGURACIÓN (EL MAPA) ---
        try:
            # Llamamos a la función que lee 'mapa.json' y construye el grafo.
            self.load_config_and_build_graphs("mapa.json")
        except FileNotFoundError:
            # Si no se encuentra el archivo, mostramos un error amigable en la GUI.
            print("ERROR: No se encontró el archivo 'mapa.json'")
            self.results_textbox.insert("0.0", "ERROR: No se encontró el archivo 'mapa.json'\n\nAsegúrate de que el archivo está en la misma carpeta.")
            self.results_textbox.configure(state="disabled") # Hacemos la caja de texto de solo lectura.
            return # Detenemos la inicialización.
        except Exception as e:
            # Si ocurre cualquier otro error al leer el JSON (ej. formato incorrecto).
            print(f"Error al cargar 'mapa.json': {e}")
            self.results_textbox.insert("0.0", f"Error al cargar 'mapa.json':\n{e}")
            self.results_textbox.configure(state="disabled")
            return

        # --- 3. AÑADIR LOS BOTONES DE DESTINO ---
        # Ahora que 'self.DESTINOS' existe (cargado del JSON), creamos los botones.
        for destino_nombre in self.DESTINOS.keys():
            # Creamos un botón para cada destino.
            button = customtkinter.CTkButton(
                self.button_frame,
                text=destino_nombre,
                # 'lambda' es crucial aquí. Crea una función "temporal" que
                # "recuerda" el 'destino_nombre' (d) de esta iteración del bucle.
                # Sin esto, todos los botones llamarían a la función con el *último* destino.
                command=lambda d=destino_nombre: self.on_button_click(d)
            )
            button.pack(side="left", padx=10, pady=10) # Los apilamos a la izquierda.

        # --- 4. DIBUJAR EL GRAFO INICIAL Y MOSTRAR MENSAJE ---
        self.draw_graph() # Llamamos a la función de dibujo por primera vez.
        self.results_textbox.insert("0.0", "Bienvenido.\n\nMapa cargado desde 'mapa.json'.\n\nEste calculador usa el algoritmo 'Doble Dijkstra'.\n\nSelecciona un destino.")
        self.results_textbox.configure(state="disabled") # Solo lectura.
    
    # --- 6. FUNCIÓN PARA CARGAR Y CONSTRUIR EL GRAFO ---
    def load_config_and_build_graphs(self, filename):
        """
        Lee el archivo JSON, extrae la configuración del mapa,
        y construye los grafos de NetworkX (self.G y self.DG).
        """
        
        # Abre el archivo JSON en modo lectura ('r') con codificación 'utf-8'.
        with open(filename, 'r', encoding='utf-8') as f:
            # Carga el contenido del archivo en un diccionario de Python.
            config = json.load(f)

        # --- Cargar datos del JSON en variables de la clase ---
        
        # 'tuple()' es importante porque los nodos de NetworkX deben ser "hashables" (listas no lo son).
        self.CASA_JAVIER = tuple(config['casa_javier'])
        self.CASA_ANDREINA = tuple(config['casa_andreina'])
        # Crea un diccionario de {NombreDestino: (calle, carrera)}
        self.DESTINOS = {name: tuple(coords) for name, coords in config['destinos'].items()}
        
        # 'range(50, 56)' crea una secuencia de 50, 51, 52, 53, 54, 55
        self.CALLES = range(config['rango_calles'][0], config['rango_calles'][1])
        self.CARRERAS = range(config['rango_carreras'][0], config['rango_carreras'][1])
        
        # Cargar los costos (tiempos) de caminata.
        COSTO_NORMAL = config['costos']['normal']
        COSTO_CARRERAS_LENTAS = config['costos']['carrera_lenta']
        COSTO_CALLE_LENTA = config['costos']['calle_lenta']
        
        # 'set()' es una optimización. Comprobar 'k in carreras_lentas'
        # es mucho más rápido si 'carreras_lentas' es un 'set' que si es una 'list'.
        carreras_lentas = set(config['reglas_costos']['carreras_lentas'])
        calles_lentas = set(config['reglas_costos']['calles_lentas'])

        # --- Construir el Grafo (el "mapa") ---
        
        # self.G es un grafo NO-DIRIGIDO (las calles son de doble sentido).
        # Lo usamos para DIBUJAR.
        self.G = nx.Graph()
        
        # 'self.pos' almacena las coordenadas (x, y) de cada nodo para el dibujo.
        self.pos = {} 
        # 'self.node_colors' almacena el color de cada nodo.
        self.node_colors = {}
        # Diccionario temporal para los colores de las aristas.
        edge_colors = {}
        
        # Doble bucle para crear cada "intersección" (nodo).
        for c in self.CALLES:
            for k in self.CARRERAS:
                node = (c, k) # El nodo es la tupla (calle, carrera)
                # La posición de dibujo (x, y) es (carrera, calle)
                # para que las carreras estén en el eje X y las calles en el Y.
                self.pos[node] = (k, c) 
                self.G.add_node(node) # Añadimos el nodo al grafo.
                self.node_colors[node] = "#CCCCCC" # Color gris por defecto.

        # Colorear nodos especiales (sobrescribe el color gris).
        self.node_colors[self.CASA_JAVIER] = "#00FF00" # Verde
        self.node_colors[self.CASA_ANDREINA] = "#FF00FF" # Magenta
        for dest in self.DESTINOS.values():
            self.node_colors[dest] = "#FFFF00" # Amarillo
                
        # --- Añadir Aristas (las "calles") y sus Pesos (costos) ---
        # Volvemos a iterar por todas las intersecciones.
        for c in self.CALLES:
            for k in self.CARRERAS:
                node = (c, k)
                
                # --- Conectar con Vecino Norte/Sur (moviéndose por una CARRERA) ---
                # Si la calle 'c-1' (ej. 53 si 'c' es 54) está en nuestro rango...
                if c - 1 in self.CALLES:
                    # Decidimos el costo:
                    # Si la CARRERA 'k' por la que nos movemos está en la lista de 'carreras_lentas'...
                    costo = COSTO_CARRERAS_LENTAS if k in carreras_lentas else COSTO_NORMAL
                    # ...usamos ese costo. Si no, el normal.
                    
                    # Añadimos la arista (la "cuadra") con su peso (costo/tiempo).
                    self.G.add_edge(node, (c - 1, k), weight=costo)
                    edge_colors[(node, (c - 1, k))] = "#AAAAAA" # Gris

                # --- Conectar con Vecino Este/Oeste (moviéndose por una CALLE) ---
                # Si la carrera 'k-1' (ej. 13 si 'k' es 14) está en nuestro rango...
                if k - 1 in self.CARRERAS:
                    # Decidimos el costo:
                    # Si la CALLE 'c' por la que nos movemos está en la lista de 'calles_lentas'...
                    costo = COSTO_CALLE_LENTA if c in calles_lentas else COSTO_NORMAL
                    # ...usamos ese costo. Si no, el normal.
                    
                    self.G.add_edge(node, (c, k - 1), weight=costo)
                    edge_colors[(node, (c, k - 1))] = "#AAAAAA" # Gris
                        
        # --- Preparar Listas de Colores para el Dibujo ---
        # NetworkX necesita que las listas de colores estén en el MISMO orden
        # que 'self.G.nodes()' y 'self.G.edges()'. Las pre-calculamos aquí.
        self.node_color_list = [self.node_colors[node] for node in self.G.nodes()]
        
        self.edge_color_list = []
        for u, v in self.G.edges():
            # Buscamos el color (NetworkX no garantiza el orden 'u, v')
            color = edge_colors.get((u, v), edge_colors.get((v, u), "#AAAAAA"))
            self.edge_color_list.append(color)
    
    # --- 7. OTRAS FUNCIONES DE LA CLASE ---
    
    def draw_graph(self, javier_path=None, andreina_path=None):
        """
        Dibuja (o redibuja) el grafo en el lienzo de Matplotlib.
        Opcionalmente, resalta las rutas de Javier y Andreína.
        """
        self.ax.clear() # Limpia el dibujo anterior.
        
        # --- Dibujar etiquetas de Calles y Carreras en los ejes ---
        for k in self.CARRERAS: self.ax.text(k, 49.7, f"K{k}", color="white", ha="center", va="top", fontsize=10)
        for c in self.CALLES: self.ax.text(9.8, c, f"C{c}", color="white", ha="right", va="center", fontsize=10)
        
        # --- Dibujar el Grafo Base ---
        nx.draw_networkx(
            self.G, self.pos, ax=self.ax,
            node_color=self.node_color_list, # Usa la lista de colores de nodos.
            edge_color=self.edge_color_list, # Usa la lista de colores de aristas.
            with_labels=False, node_size=250 # No dibujar etiquetas feas (ej. "(54, 14)")
        )
        
        # --- Dibujar Etiquetas de Costos (Pesos) ---
        edge_labels = nx.get_edge_attributes(self.G, 'weight') # Obtiene todos los pesos.
        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, ax=self.ax,
            font_color='white', font_size=8,
            # 'bbox' crea un pequeño fondo para que el número sea legible.
            bbox=dict(facecolor=self.ax.get_facecolor(), edgecolor='none', pad=0.2, alpha=0.8)
        )
        
        # --- Dibujar Etiquetas de Nodos Especiales ---
        labels = {
            self.CASA_JAVIER: "Javier", 
            self.CASA_ANDREINA: "Andreína",
            # Desempaqueta el diccionario de destinos para añadir "The Darkness", etc.
            **{dest: name for name, dest in self.DESTINOS.items()} 
        }
        nx.draw_networkx_labels(
            self.G, self.pos, labels=labels, ax=self.ax, 
            font_color='white', font_size=10
        )
        
        # --- Resaltar Rutas (si se proporcionan) ---
        if javier_path:
            # 'zip' convierte [A, B, C] en [(A, B), (B, C)]
            path_edges = list(zip(javier_path, javier_path[1:]))
            # Vuelve a dibujar SÓLO las aristas de la ruta, con un color y grosor diferentes.
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax, edgelist=path_edges,
                edge_color="#00FFFF", width=3.0 # Cyan
            )
        if andreina_path:
            path_edges = list(zip(andreina_path, andreina_path[1:]))
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax, edgelist=path_edges,
                edge_color="#FF00FF", width=3.0 # Magenta
            )
            
        self.ax.set_axis_off() # Oculta los ejes X e Y (se ven feos).
        self.canvas.draw() # Refresca el lienzo en la GUI para mostrar los cambios.


    def get_path_cost_from_original_graph(self, path):
        """
        Calcula el costo de una ruta dada usando el grafo original (self.G).
        Esto es una verificación, ya que Dijkstra calcula el costo en grafos
        temporales (G_temp1, G_temp2) que podrían tener aristas faltantes.
        """
        cost = 0
        # Itera sobre la ruta como pares de (A, B), (B, C), ...
        for u, v in zip(path, path[1:]):
            if self.G.has_edge(u, v):
                # Suma el peso de la arista del grafo ORIGINAL.
                cost += self.G[u][v]['weight']
            else:
                # Si la arista no existe (no debería pasar), es un error.
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
        
        # Usamos self.G (el grafo no-dirigido)
        ruta_j1, tiempo_j1 = find_shortest_path_dijkstra(self.G, self.CASA_JAVIER, destino_coords) # <-- CAMBIO
        
        if tiempo_j1 == float('inf'):
            output += "ERROR: Javier no puede llegar al destino.\n"
            self.results_textbox.insert("0.0", output)
            self.results_textbox.configure(state="disabled")
            return

        # Copiamos el grafo no-dirigido
        G_temp1 = self.G.copy()
        
        # Lógica de borrado (solo 1 línea)
        for u, v in zip(ruta_j1, ruta_j1[1:]):
            if G_temp1.has_edge(u, v): 
                G_temp1.remove_edge(u, v) # (borra la calle entera)
        
        # Calculamos la ruta de Andreína en el grafo no-dirigido modificado
        ruta_a1, tiempo_a1 = find_shortest_path_dijkstra(G_temp1, self.CASA_ANDREINA, destino_coords)
        
        total_tiempo1 = tiempo_j1 + tiempo_a1
        output += f"  Ruta Javier: {tiempo_j1} min\n"
        output += f"  Ruta Andreína: {tiempo_a1} min\n"
        output += f"  Tiempo Total Escenario 1: {total_tiempo1} min\n\n"

        # --- ESCENARIO 2: ANDREÍNA ELIGE PRIMERO ---
        output += "--- Escenario 2: Andreína elige primero ---\n"
        
        # Usamos self.G (el grafo no-dirigido)
        ruta_a2, tiempo_a2 = find_shortest_path_dijkstra(self.G, self.CASA_ANDREINA, destino_coords) # <-- CAMBIO

        if tiempo_a2 == float('inf'):
            output += "ERROR: Andreína no puede llegar al destino.\n"
            self.results_textbox.insert("0.0", output)
            self.results_textbox.configure(state="disabled")
            return

        # Copiamos el grafo no-dirigido
        G_temp2 = self.G.copy()
        
        # Lógica de borrado simplificada (solo 1 línea)
        for u, v in zip(ruta_a2, ruta_a2[1:]):
            if G_temp2.has_edge(u, v): 
                G_temp2.remove_edge(u, v) # (borra la calle entera)
        
        # Calculamos la ruta de Javier en el grafo no-dirigido
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
            # Si Javier es más lento, él sale primero. Andreina espera.
            output += f"  Javier sale primero.\n"
            output += f"  Andreina debe salir {diferencia} minutos DESPUÉS de Javier.\n"
        elif tiempo_a_final > tiempo_j_final:
            diferencia = tiempo_a_final - tiempo_j_final
            # Si Andreina es más lenta, ella sale primero. Javier espera.
            output += f"  Andreina sale primero.\n"
            output += f"  Javier debe salir {diferencia} minutos DESPUÉS de Andreina.\n"
        else:
            output += "  Ambos tienen el mismo tiempo. Pueden salir juntos.\n"
            
        self.draw_graph(javier_path=ruta_j_final, andreina_path=ruta_a_final)
        
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", output)
        self.results_textbox.configure(state="disabled")

# --- 8. Punto de Entrada Principal ---
# Este bloque solo se ejecuta si corres el script directamente
if __name__ == "__main__":
    app = App() # Crea una instancia de nuestra clase de aplicación.
    app.mainloop() # Inicia el bucle de la GUI (espera clics, etc.)