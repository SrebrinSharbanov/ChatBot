# PowerShell скрипт за автоматично setup на D:\ директории
# Стартирай като Administrator: .\scripts\setup-d-drive.ps1

Write-Host "=== Mini RAG Chatbot - D:\ Drive Setup ===" -ForegroundColor Cyan
Write-Host ""

# Проверка дали D:\ съществува
if (-not (Test-Path "D:\")) {
    Write-Host "ERROR: D:\ drive не съществува!" -ForegroundColor Red
    exit 1
}

# Провери свободно място на D:\
$drive = Get-PSDrive -Name D
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "Свободно място на D:\: $freeSpaceGB GB" -ForegroundColor Yellow

if ($freeSpaceGB -lt 20) {
    Write-Host "WARNING: Препоръчва се поне 20 GB свободно място!" -ForegroundColor Yellow
    $continue = Read-Host "Продължи така или така? (y/n)"
    if ($continue -ne "y") {
        exit 0
    }
}

Write-Host ""
Write-Host "Създаване на директории на D:\..." -ForegroundColor Green

# Създай основна структура
$directories = @(
    "D:\mini-rag-chatbot",
    "D:\mini-rag-chatbot\data",
    "D:\mini-rag-chatbot\data\postgres",
    "D:\mini-rag-chatbot\data\embeddings",
    "D:\mini-rag-chatbot\models",
    "D:\mini-rag-chatbot\logs",
    "D:\DockerData",
    "D:\ollama",
    "D:\ollama\models"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✓ Създадена: $dir" -ForegroundColor Green
    } else {
        Write-Host "✓ Съществува: $dir" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Копиране на .env файл..." -ForegroundColor Green

# Копирай .env.example към .env ако не съществува
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env файлът е създаден от .env.example" -ForegroundColor Green
    Write-Host "  Моля, редактирай .env файла ако е необходимо" -ForegroundColor Yellow
} else {
    Write-Host "✓ .env файлът вече съществува" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Настройка на Ollama environment variable..." -ForegroundColor Green

# Задай OLLAMA_MODELS environment variable
try {
    [System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', 'D:\ollama\models', 'User')
    Write-Host "✓ OLLAMA_MODELS е зададена на D:\ollama\models" -ForegroundColor Green
    Write-Host "  ВАЖНО: Рестартирай Ollama service за да се приложи промяната" -ForegroundColor Yellow
} catch {
    Write-Host "✗ Грешка при задаване на OLLAMA_MODELS" -ForegroundColor Red
}

Write-Host ""
Write-Host "Проверка на Docker конфигурация..." -ForegroundColor Green

# Провери Docker daemon.json
$dockerConfigPath = "C:\ProgramData\Docker\config\daemon.json"
$dockerConfigDir = Split-Path $dockerConfigPath -Parent

if (Test-Path "C:\Program Files\Docker\Docker\Docker Desktop.exe") {
    Write-Host "✓ Docker Desktop е намерен" -ForegroundColor Green
    
    if (-not (Test-Path $dockerConfigPath)) {
        Write-Host ""
        Write-Host "Docker daemon.json не съществува. Искаш ли да го създам? (y/n)" -ForegroundColor Yellow
        $createConfig = Read-Host
        
        if ($createConfig -eq "y") {
            if (-not (Test-Path $dockerConfigDir)) {
                New-Item -ItemType Directory -Path $dockerConfigDir -Force | Out-Null
            }
            
            Copy-Item "config\docker-daemon.json" $dockerConfigPath -Force
            Write-Host "✓ Docker daemon.json е създаден" -ForegroundColor Green
            Write-Host "  ВАЖНО: Рестартирай Docker Desktop за да се приложи промяната" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✓ Docker daemon.json вече съществува" -ForegroundColor Gray
        Write-Host "  Провери дали data-root е зададен на D:\DockerData" -ForegroundColor Yellow
    }
} else {
    Write-Host "! Docker Desktop не е намерен. Инсталирай го от docker.com" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup завърши успешно! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Следващи стъпки:" -ForegroundColor Yellow
Write-Host "1. Прочети DOCKER_SETUP.md за детайлни инструкции" -ForegroundColor White
Write-Host "2. Рестартирай Docker Desktop (ако променяш daemon.json)" -ForegroundColor White
Write-Host "3. Рестартирай Ollama service (за OLLAMA_MODELS)" -ForegroundColor White
Write-Host "4. Провери конфигурацията: docker info | Select-String 'Docker Root Dir'" -ForegroundColor White
Write-Host "5. Стартирай проекта: docker-compose up -d" -ForegroundColor White
Write-Host ""

# Покажи структурата
Write-Host "Структура на D:\ диск:" -ForegroundColor Cyan
tree D:\mini-rag-chatbot /F /A
Write-Host ""

