# Analisis de Inventario

Dashboard web para analizar inventario por bodega, linea, categoria y proveedor. Genera informes con IA y exporta reportes en Word/Excel.

## Ejecutar en local

```bash
pip install -r requirements.txt
```

Crear archivo `.env` con tu API key de Gemini:
```
GEMINI_API_KEY=tu_api_key_aqui
```

Ejecutar:
```bash
uvicorn app:app --reload --port 8001
```

Abrir http://127.0.0.1:8001

## Deploy en Render

1. Crear cuenta en [render.com](https://render.com)
2. Conectar el repositorio de GitHub
3. Configurar la variable de entorno `GEMINI_API_KEY` en el dashboard de Render
4. Deploy automatico al hacer push

## Funcionalidades

- Carga archivos Excel de inventario
- Pivot table automatica por bodega
- Metricas por bodega: existencia, comprometida, disponible, valor, margen
- Analisis por linea, sub-linea, categoria, canal, proveedor, estado
- Deteccion de items sin movimiento y alertas de stock
- Generacion de informes con IA (Gemini, Claude, ChatGPT)
- Exportacion a Word (.docx) y Excel (.xlsx)
