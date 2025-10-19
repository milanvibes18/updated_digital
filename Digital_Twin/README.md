# 🏭 Digital Twin System

A comprehensive Digital Twin platform for industrial IoT monitoring, predictive analytics, and real-time system health management.

## 🌟 Features

- **Real-time Monitoring**: Live data streaming from IoT devices
- **Predictive Analytics**: AI-powered anomaly detection and risk prediction
- **Interactive Dashboard**: Modern web interface with real-time visualizations
- **Security**: Enterprise-grade encryption and secure data management
- **Scalable Architecture**: Microservices-ready with Docker and Kubernetes support
- **Health Scoring**: Automated system health assessment and scoring
- **Alert Management**: Intelligent alerting system with customizable thresholds

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT Devices   │────│  Data Gateway   │────│  Digital Twin   │
└─────────────────┘    └─────────────────┘    │    Platform     │
                                              └─────────────────┘
                                                       │
                       ┌─────────────────┐────────────┘
                       │  AI Analytics   │
                       │     Engine      │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Node.js (for frontend development)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Digital_Twin
   ```

2. **Set up Python environment**
   ```bash
   conda env create -f environment.yml
   conda activate digital_twin
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   python DATA_MANAGEMENT/unified_data_generator.py
   ```

5. **Run the application**
   ```bash
   python WEB_APPLICATION/enhanced_flask_app_v2.py
   ```

### Using Docker

```bash
docker-compose up -d
```

## 📊 Dashboard Access

- **Main Dashboard**: http://localhost:5000
- **Analytics**: http://localhost:5000/analytics
- **Device Management**: http://localhost:5000/devices
- **API Documentation**: http://localhost:5000/api/docs

## 🔧 Configuration

### Environment Variables

```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
export DATABASE_URL=sqlite:///DATABASE/health_data.db
export ENCRYPTION_KEY_PATH=CONFIG/encryption.key
```

### System Configuration

Edit `CONFIG/system_config.json` to customize:
- Alert thresholds
- Health score weights
- Data collection intervals
- Security settings

## 🔒 Security Features

- **Data Encryption**: AES-256 encryption for sensitive data
- **Secure Authentication**: JWT-based authentication
- **Audit Logging**: Comprehensive security audit trails
- **Role-based Access**: Granular permission system
- **Data Backup**: Automated encrypted backups

## 🤖 AI Modules

### Predictive Analytics
- **Anomaly Detection**: Isolation Forest and statistical methods
- **Risk Prediction**: Machine learning-based risk assessment
- **Pattern Analysis**: Time series pattern recognition
- **Health Scoring**: Multi-factor health assessment

### Real-time Processing
- **WebSocket Integration**: Live data streaming
- **Alert Management**: Intelligent threshold-based alerting
- **Dashboard Updates**: Real-time visualization updates

## 📈 Monitoring & Analytics

### Key Metrics
- System health scores
- Device performance indicators
- Predictive maintenance alerts
- Energy efficiency metrics
- Security event tracking

### Visualization
- Real-time charts and graphs
- Historical trend analysis
- Comparative analytics
- Custom dashboards

## 🛠️ Development

### Project Structure
```
Digital_Twin/
├── AI_MODULES/          # AI and analytics components
├── WEB_APPLICATION/     # Flask web application
├── CONFIG/             # Configuration files
├── DATA_MANAGEMENT/    # Data processing and management
├── SECURITY/           # Security and encryption
├── TESTS/              # Unit and integration tests
└── DEPLOYMENT/         # Docker and Kubernetes configs
```

### Running Tests
```bash
python -m pytest TESTS/
```

### Code Quality
```bash
flake8 --max-line-length=100
black --line-length=100 .
```

## 🚀 Deployment

### Docker Deployment
```bash
docker build -t digital-twin .
docker run -p 5000:5000 digital-twin
```

### Kubernetes Deployment
```bash
kubectl apply -f DEPLOYMENT/kubernetes/
```

### Cloud Deployment
- **AWS**: See `DEPLOYMENT/cloud/aws/`
- **Azure**: See `DEPLOYMENT/cloud/azure/`
- **GCP**: See `DEPLOYMENT/cloud/gcp/`

## 📚 Documentation

- [API Documentation](DOCUMENTATION/API_DOCUMENTATION.md)
- [Architecture Guide](DOCUMENTATION/ARCHITECTURE.md)
- [Security Guide](DOCUMENTATION/SECURITY_GUIDE.md)
- [Deployment Guide](DOCUMENTATION/DEPLOYMENT_GUIDE.md)

## 🔧 Maintenance

### Regular Tasks
- Database cleanup: `python DATA_MANAGEMENT/auto_cleanup.py`
- Log rotation: Automated via system configuration
- Security audits: Weekly automated scans
- Performance monitoring: Real-time metrics

### Troubleshooting

#### Common Issues
1. **Database Connection**: Check database file permissions
2. **WebSocket Issues**: Verify port 5000 availability
3. **Encryption Errors**: Ensure encryption keys are generated
4. **Memory Usage**: Monitor system resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: See DOCUMENTATION/ folder
- **Community**: [Discussion Forum]

## 🔄 Version History

- **v2.0**: Enhanced AI modules and security features
- **v1.5**: WebSocket integration and real-time updates
- **v1.0**: Initial release with basic monitoring

---

**Built with ❤️ for Industrial IoT and Digital Twin Applications**