# Usa una imagen oficial de Python como imagen base
FROM python:3.9-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo de dependencias al contenedor (esto ayuda a aprovechar la cache de Docker)
COPY requirements.txt .

# Instala las dependencias necesarias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código fuente de la aplicación al contenedor (directorio de trabajo)   
COPY . .

#comando por defecto que se ejecuta al iniciar el contenedor
# (lo sobreescribe al ejecutar el contenedor)
CMD ["python", "src/train.py"]
