# 🚀 Deployment Guide - Drone Emergency Detection Dashboard

## Quick Deploy Options

### 1. **Heroku (Recommended - Free)**
```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login to Heroku
heroku login

# Create new app
heroku create your-dashboard-name

# Add buildpack for OpenCV
heroku buildpacks:add --index 1 https://github.com/heroku/heroku-buildpack-apt

# Add Python buildpack
heroku buildpacks:add --index 2 heroku/python

# Create Aptfile for OpenCV dependencies
echo "libgl1-mesa-glx" > Aptfile
echo "libglib2.0-0" >> Aptfile

# Deploy
git add .
git commit -m "Deploy dashboard"
git push heroku main

# Open the app
heroku open
```

### 2. **Railway (Simple - Free)**
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway will auto-detect Python and deploy
4. Add environment variables if needed

### 3. **Render (Good Free Tier)**
1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Set build command: `pip install -r requirements_production.txt`
5. Set start command: `gunicorn --bind 0.0.0.0:$PORT drone_dashboard:app`

## Environment Variables

Add these to your deployment platform:

```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
```

## Model Files

**Important**: You need to upload your trained model files to the deployment platform:

1. **For Heroku**: Use Heroku S3 or similar for model storage
2. **For Railway/Render**: Include model files in repository (if under size limits)

### Model Upload Options:

#### Option A: Include in Repository (Small models only)
```bash
# Add model files to git (if under 100MB)
git add models/car_crash_detection2/weights/best.pt
git commit -m "Add model files"
```

#### Option B: Cloud Storage (Recommended)
1. Upload models to AWS S3, Google Cloud Storage, or similar
2. Modify `load_detection_model()` to download from cloud storage
3. Add cloud storage credentials as environment variables

## Production Considerations

### 1. **Database**
- SQLite works for small deployments
- For production: Use PostgreSQL or MySQL
- Consider using cloud databases (Heroku Postgres, Railway Postgres)

### 2. **File Storage**
- Detection images should be stored in cloud storage
- Use AWS S3, Google Cloud Storage, or similar
- Don't store files on ephemeral filesystem

### 3. **Security**
- Change default secret keys
- Add authentication if needed
- Use HTTPS in production
- Consider rate limiting

### 4. **Performance**
- Add caching (Redis)
- Optimize model loading
- Use CDN for static files
- Consider load balancing for high traffic

## Monitoring

### 1. **Logs**
```bash
# Heroku
heroku logs --tail

# Railway
railway logs

# Render
# Available in dashboard
```

### 2. **Health Checks**
Add a health check endpoint:
```python
@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})
```

## Custom Domain

### Heroku
```bash
heroku domains:add yourdomain.com
```

### Railway/Render
- Add custom domain in dashboard
- Update DNS records

## SSL/HTTPS
- Heroku: Automatic
- Railway: Automatic
- Render: Automatic

## Troubleshooting

### Common Issues:

1. **Port Issues**
   - Make sure to use `$PORT` environment variable
   - Don't hardcode port 5000

2. **Model Loading**
   - Check model file paths
   - Ensure models are included in deployment

3. **Database Issues**
   - SQLite may not work on some platforms
   - Use cloud databases for production

4. **Memory Issues**
   - YOLO models can be memory-intensive
   - Consider using smaller models
   - Add memory limits in deployment config

## Scaling

### Horizontal Scaling
- Use multiple instances
- Add load balancer
- Use shared database

### Vertical Scaling
- Increase memory/CPU
- Use more powerful instances

## Cost Optimization

### Free Tiers:
- **Heroku**: $0/month (sleeps after 30min inactivity)
- **Railway**: $0/month (limited usage)
- **Render**: $0/month (sleeps after 15min inactivity)

### Paid Options:
- **Heroku**: $7/month (Basic dyno)
- **Railway**: $5/month (Pro plan)
- **Render**: $7/month (Standard plan)

## Support

For deployment issues:
1. Check platform-specific documentation
2. Review logs for errors
3. Test locally with production settings
4. Consider using Docker for consistent environments 