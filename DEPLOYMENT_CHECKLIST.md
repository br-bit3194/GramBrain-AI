# GramBrain AI - Deployment Checklist

## Pre-Deployment Verification

### Backend Verification
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Backend starts without errors: `python -m uvicorn src.api.routes:app --reload`
- [ ] Health check works: `curl http://localhost:8000/health`
- [ ] All endpoints respond with proper JSON
- [ ] Validation errors return proper format (not objects)
- [ ] CORS headers present in responses
- [ ] No Python errors in logs
- [ ] All 12 agents listed in health check

### Frontend Verification
- [ ] All dependencies installed: `npm install`
- [ ] Frontend starts without errors: `npm run dev`
- [ ] Home page loads at `http://localhost:3000`
- [ ] No console errors
- [ ] All pages accessible
- [ ] API client configured correctly
- [ ] Environment variables set in `.env.local`
- [ ] No TypeScript errors

### Integration Verification
- [ ] Backend and frontend can communicate
- [ ] User registration works end-to-end
- [ ] No CORS errors
- [ ] No validation error objects in UI
- [ ] Success messages display correctly
- [ ] Error messages display correctly
- [ ] All API endpoints tested
- [ ] All pages tested

### Testing Verification
- [ ] Backend tests pass: `pytest tests/`
- [ ] Frontend builds without errors: `npm run build`
- [ ] No console warnings
- [ ] No console errors
- [ ] All features work as expected

---

## Production Deployment

### Environment Setup

#### Backend (.env)
```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/grambrain

# AWS Bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Vector DB
VECTOR_DB_URL=http://vector-db:6379

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Logging
LOG_LEVEL=INFO
```

#### Frontend (.env.production)
```bash
NEXT_PUBLIC_API_URL=https://api.grambrain.com/api
NEXT_PUBLIC_APP_NAME=GramBrain AI
NEXT_PUBLIC_APP_VERSION=1.0.0
```

### Docker Deployment

#### Build Images
```bash
# Backend
docker build -t grambrain-backend:latest ./backend

# Frontend
docker build -t grambrain-frontend:latest ./frontend
```

#### Run with Docker Compose
```bash
docker-compose up -d
```

#### Verify Containers
```bash
docker ps
docker logs grambrain-backend
docker logs grambrain-frontend
```

### Kubernetes Deployment (Optional)

#### Create Deployments
```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/services.yaml
```

#### Verify Deployments
```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

---

## Post-Deployment Verification

### Health Checks
- [ ] Backend health endpoint responds: `curl https://api.grambrain.com/health`
- [ ] Frontend loads: `https://grambrain.com`
- [ ] All API endpoints accessible
- [ ] Database connected
- [ ] Vector DB connected
- [ ] AWS Bedrock accessible

### Monitoring Setup
- [ ] Application logs configured
- [ ] Error tracking enabled (Sentry, etc.)
- [ ] Performance monitoring enabled (New Relic, etc.)
- [ ] Uptime monitoring configured
- [ ] Alert notifications set up

### Security Checks
- [ ] HTTPS enabled
- [ ] CORS properly configured
- [ ] API authentication implemented
- [ ] Rate limiting enabled
- [ ] Input validation working
- [ ] Error messages don't leak sensitive info
- [ ] Environment variables not exposed
- [ ] Database credentials secured

### Performance Checks
- [ ] Page load time < 3 seconds
- [ ] API response time < 500ms
- [ ] Database queries optimized
- [ ] Caching implemented
- [ ] CDN configured (if applicable)
- [ ] Images optimized
- [ ] Code minified

### Backup & Recovery
- [ ] Database backups configured
- [ ] Backup retention policy set
- [ ] Disaster recovery plan documented
- [ ] Rollback procedure tested
- [ ] Data export capability verified

---

## Scaling Considerations

### Horizontal Scaling
- [ ] Load balancer configured
- [ ] Multiple backend instances
- [ ] Multiple frontend instances
- [ ] Session management (if needed)
- [ ] Database replication

### Vertical Scaling
- [ ] CPU allocation reviewed
- [ ] Memory allocation reviewed
- [ ] Storage allocation reviewed
- [ ] Network bandwidth reviewed

### Database Scaling
- [ ] Connection pooling configured
- [ ] Query optimization done
- [ ] Indexing optimized
- [ ] Partitioning considered
- [ ] Read replicas configured

---

## Maintenance Plan

### Daily Tasks
- [ ] Monitor error logs
- [ ] Check system health
- [ ] Verify backups completed
- [ ] Monitor performance metrics

### Weekly Tasks
- [ ] Review security logs
- [ ] Check for updates
- [ ] Verify backup integrity
- [ ] Performance analysis

### Monthly Tasks
- [ ] Security audit
- [ ] Dependency updates
- [ ] Database maintenance
- [ ] Capacity planning

### Quarterly Tasks
- [ ] Full system audit
- [ ] Disaster recovery drill
- [ ] Performance optimization
- [ ] Security penetration testing

---

## Rollback Procedure

### If Deployment Fails

1. **Identify Issue**
   ```bash
   docker logs grambrain-backend
   docker logs grambrain-frontend
   ```

2. **Rollback to Previous Version**
   ```bash
   docker-compose down
   docker pull grambrain-backend:previous
   docker pull grambrain-frontend:previous
   docker-compose up -d
   ```

3. **Verify Rollback**
   ```bash
   curl https://api.grambrain.com/health
   ```

4. **Investigate Issue**
   - Check logs
   - Review changes
   - Fix issues
   - Test in staging

---

## Documentation Updates

- [ ] Update API documentation
- [ ] Update deployment guide
- [ ] Update troubleshooting guide
- [ ] Update user documentation
- [ ] Update admin documentation
- [ ] Update developer documentation

---

## Sign-Off

- [ ] Backend Lead: _________________ Date: _______
- [ ] Frontend Lead: ________________ Date: _______
- [ ] DevOps Lead: _________________ Date: _______
- [ ] QA Lead: ____________________ Date: _______
- [ ] Project Manager: _____________ Date: _______

---

## Post-Deployment Notes

```
Date Deployed: _______________
Version: _______________
Deployed By: _______________
Issues Encountered: _______________
Resolution: _______________
Performance Metrics: _______________
Next Steps: _______________
```

---

## Emergency Contacts

- Backend Support: _______________
- Frontend Support: _______________
- DevOps Support: _______________
- Database Support: _______________
- Security Team: _______________

---

## Useful Commands

### Docker
```bash
# View logs
docker logs -f grambrain-backend
docker logs -f grambrain-frontend

# Restart services
docker-compose restart

# Stop services
docker-compose down

# View resource usage
docker stats
```

### Kubernetes
```bash
# View logs
kubectl logs -f deployment/grambrain-backend
kubectl logs -f deployment/grambrain-frontend

# Restart deployment
kubectl rollout restart deployment/grambrain-backend

# View events
kubectl get events

# Describe pod
kubectl describe pod <pod-name>
```

### Database
```bash
# Connect to database
psql -h host -U user -d grambrain

# Backup database
pg_dump -h host -U user grambrain > backup.sql

# Restore database
psql -h host -U user grambrain < backup.sql
```

---

## Success Criteria

✅ All health checks pass
✅ All endpoints respond correctly
✅ No errors in logs
✅ Performance metrics acceptable
✅ Security checks passed
✅ Backups verified
✅ Monitoring configured
✅ Documentation updated
✅ Team trained
✅ Rollback procedure tested

**Status: READY FOR PRODUCTION** 🚀

