# Docker Setup за D:\ Диск

## Проблем
По подразбиране Docker Desktop на Windows използва C:\ диск за съхранение на images, containers и volumes. Този setup пренасочва всичко към D:\ диск.

## Решение

### 1. Преместване на Docker Data Root (Препоръчително)

#### Вариант A: Чрез Docker Desktop Settings (Лесен начин)
1. Отвори **Docker Desktop**
2. Отиди на **Settings** (иконка със зъбчатка)
3. Избери **Resources** → **Advanced**
4. Промени **Disk image location** на: `D:\DockerData`
5. Кликни **Apply & Restart**

#### Вариант B: Чрез daemon.json (Ръчна конфигурация)
1. Спри Docker Desktop
2. Копирай `config/docker-daemon.json` в:
   - `C:\ProgramData\Docker\config\daemon.json` (ако няма такъв файл, създай го)
3. Създай директорията: `D:\DockerData`
4. Стартирай Docker Desktop

### 2. Преместване на WSL2 Data (Ако използваш WSL2 backend)

Ако Docker Desktop използва WSL2 backend, трябва да преместиш и WSL2 дистрибуцията:

```powershell
# 1. Спри Docker Desktop и WSL
wsl --shutdown

# 2. Експортирай docker-desktop-data
wsl --export docker-desktop-data D:\WSL\docker-desktop-data.tar

# 3. Дерегистрирай старата дистрибуция
wsl --unregister docker-desktop-data

# 4. Импортирай на новата локация
wsl --import docker-desktop-data D:\WSL\docker-desktop-data D:\WSL\docker-desktop-data.tar --version 2

# 5. Направи същото за docker-desktop (ако е необходимо)
wsl --export docker-desktop D:\WSL\docker-desktop.tar
wsl --unregister docker-desktop
wsl --import docker-desktop D:\WSL\docker-desktop D:\WSL\docker-desktop.tar --version 2

# 6. Изтрий временните tar файлове
Remove-Item D:\WSL\docker-desktop-data.tar
Remove-Item D:\WSL\docker-desktop.tar

# 7. Стартирай Docker Desktop
```

### 3. Docker Compose Volumes за Проекта

В `docker-compose.yml` файла volumes са конфигурирани да използват:
```yaml
volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      device: D:/mini-rag-chatbot/data/postgres
      o: bind
```

## Структура на D:\ след настройка

```
D:\
├── DockerData/                    # Docker root data (images, containers)
│   ├── containers/
│   ├── image/
│   ├── volumes/
│   └── ...
├── WSL/                          # WSL2 дистрибуции (ако се използват)
│   ├── docker-desktop/
│   └── docker-desktop-data/
└── mini-rag-chatbot/             # Проектни данни
    ├── data/
    │   ├── postgres/             # PostgreSQL volume
    │   ├── models/               # Локални модели
    │   └── embeddings/           # Кеширани embeddings
    └── logs/                     # Логове
```

## Проверка на Конфигурацията

След настройката провери дали Docker използва D:\ диск:

```powershell
# Провери Docker info
docker info | Select-String "Docker Root Dir"

# Трябва да покаже: Docker Root Dir: D:\DockerData
```

## Освобождаване на Място от C:\

След успешна миграция, можеш да изтриеш старите данни:

```powershell
# ВНИМАНИЕ: Изтрий САМО ако си сигурен, че всичко работи на D:\
# Remove-Item -Recurse -Force "C:\ProgramData\Docker"
# Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Docker"
```

## Забележки

- **Disk Space**: Уверете се, че D:\ има поне 20-30 GB свободно място за Docker и моделите
- **Performance**: D:\ трябва да бъде физически диск (не USB), за да има добра performance
- **Backup**: Направете backup на важни containers/volumes преди миграция
- **Models**: Ollama моделите също ще се съхраняват на D:\ след настройката

## Ollama Data Location

Ollama по подразбиране също използва C:\. За да го промените:

```powershell
# Задай environment variable за Ollama
[System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', 'D:\ollama\models', 'User')

# Или добави в System Properties → Environment Variables
# Variable: OLLAMA_MODELS
# Value: D:\ollama\models
```

Рестартирай Ollama service след промяната.

