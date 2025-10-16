# Trabajo Final: Visión-Fauna

**Visión artificial aplicada a la automatización del análisis de videos de cámaras trampa en la selva misionera**

Trabajo Final de la Carrera de Especialización en Inteligencia Artificial (CEIA) del Laboratorio de Sistemas Embebidos (LSE) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FI-UBA)

Este repositorio contiene el código fuente y la documentación del proyecto final para desarrollar una herramienta de software que utiliza Inteligencia Artificial (IA) para automatizar el análisis de videos de fauna silvestre capturados por cámaras trampa en la Selva Paranaense de Misiones.

---

## El Problema a Resolver

El monitoreo de fauna con cámaras trampa genera una cantidad masiva de datos en video. El análisis manual de este material es ineficiente y presenta varios problemas críticos que este proyecto busca mitigar:

-   **Retrasos en la obtención de información:** El análisis puede demorar meses, lo que inutiliza la información para la toma de decisiones urgentes en conservación.
-   **Alto consumo de recursos:** Desvía a personal científico especializado de tareas de investigación hacia el filtrado manual y tedioso de videos.
-   **Errores e inconsistencia:** La fatiga y la subjetividad pueden llevar a que se omitan o clasifiquen erróneamente animales, especialmente especies pequeñas, nocturnas o parcialmente ocultas.

---

## La Solución Propuesta

Se propone una **aplicación de escritorio para Windows** que automatiza todo el flujo de trabajo. Esta herramienta está diseñada para operar de forma local en el equipo del investigador, garantizando la privacidad de los datos y permitiendo su uso en zonas con conectividad limitada.

### Características Principales

-   **Procesamiento Local:** Selección de una carpeta de videos para ser procesados directamente en la máquina del usuario.
-   **Extracción Inteligente:** El software analizará cada video, extraerá fotogramas clave y los pasará por un modelo de IA.
-   **Identificación y Agrupación:** El sistema identificará los fotogramas que contienen fauna y los agrupará por similitud, permitiendo al investigador revisar rápidamente los eventos de interés.
-   **Interfaz Gráfica Intuitiva:** Una GUI construida que guiará al usuario a través del proceso de análisis.

---