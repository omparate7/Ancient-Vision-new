import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Chip,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Palette,
  Restore,
  Memory,
  Speed,
  Info,
  CheckCircle,
  RadioButtonUnchecked
} from '@mui/icons-material';

const ModelSelection = ({ onModelSelect, currentModule = 0 }) => {
  const [artModels, setArtModels] = useState({});
  const [statueModels, setStatueModels] = useState({});
  const [selectedArtModel, setSelectedArtModel] = useState('');
  const [selectedStatueModel, setSelectedStatueModel] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [modelStatus, setModelStatus] = useState({});

  useEffect(() => {
    loadAvailableModels();
    checkModelStatus();
  }, []);

  const loadAvailableModels = async () => {
    try {
      setLoading(true);
      
      // Load art transformation models
      const artResponse = await fetch('http://localhost:5001/api/models');
      const artData = await artResponse.json();
      
      // Load statue restoration models (check availability)
      const statueResponse = await fetch('http://localhost:5002/api/statue-restoration/status');
      const statueData = await statueResponse.json();
      
      setArtModels(artData.models || {});
      
      // Create statue models info
      if (statueData.success) {
        setStatueModels({
          'statue_restoration_v1': {
            name: 'Ancient Statue Restoration',
            type: 'inpainting',
            description: 'AI-powered restoration for damaged statues',
            device: statueData.status.device,
            weights_exist: statueData.status.weights_exist,
            lazy_loading: statueData.status.lazy_loading_enabled
          }
        });
      }
      
    } catch (err) {
      console.error('Failed to load models:', err);
      setError('Failed to load available models');
    } finally {
      setLoading(false);
    }
  };

  const checkModelStatus = async () => {
    try {
      // Check art transformation status
      const artHealth = await fetch('http://localhost:5001/api/health');
      const artHealthData = await artHealth.json();
      
      // Check statue restoration status
      const statueHealth = await fetch('http://localhost:5002/api/statue-restoration/health');
      const statueHealthData = await statueHealth.json();
      
      setModelStatus({
        art: {
          loaded: artHealthData.models_loaded || false,
          lazy_loading: artHealthData.lazy_loading || false,
          device: artHealthData.device
        },
        statue: {
          loaded: statueHealthData.pipeline_status === 'loaded',
          lazy_loading: statueHealthData.lazy_loading || false,
          device: statueHealthData.device
        }
      });
      
    } catch (err) {
      console.error('Failed to check model status:', err);
    }
  };

  const handleModelSelection = (modelType, modelId) => {
    if (modelType === 'art') {
      setSelectedArtModel(modelId);
    } else {
      setSelectedStatueModel(modelId);
    }
    
    onModelSelect({
      type: modelType,
      modelId: modelId,
      model: modelType === 'art' ? artModels[modelId] : statueModels[modelId]
    });
  };

  if (loading) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="center" p={3}>
            <CircularProgress sx={{ mr: 2 }} />
            <Typography>Loading available models...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert severity="error" action={
        <Button color="inherit" size="small" onClick={loadAvailableModels}>
          Retry
        </Button>
      }>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Model Selection Header */}
      <Card elevation={3} sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
            <Memory sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6">Model Selection - Lazy Loading</Typography>
            <Tooltip title="Models load only when transformations start for optimal performance">
              <IconButton size="small" sx={{ ml: 1 }}>
                <Info fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              🚀 <strong>Performance Optimized:</strong> Models load on-demand when you start transformations. 
              Select your preferred model below - it will load automatically when needed.
            </Typography>
          </Alert>
        </CardContent>
      </Card>

      {/* Art Transformation Models */}
      {currentModule === 0 && (
        <Card elevation={3} sx={{ mb: 3 }}>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <Palette sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h6">Traditional Art Transformation</Typography>
              <Chip 
                label={modelStatus.art?.lazy_loading ? "Lazy Loading" : "Standard Loading"} 
                color="primary" 
                size="small" 
                sx={{ ml: 2 }}
              />
            </Box>

            <Grid container spacing={2}>
              {Object.entries(artModels).map(([id, model]) => (
                <Grid item xs={12} sm={6} md={4} key={id}>
                  <Card 
                    variant={selectedArtModel === id ? "outlined" : "elevation"}
                    sx={{ 
                      cursor: 'pointer',
                      border: selectedArtModel === id ? 2 : 1,
                      borderColor: selectedArtModel === id ? 'primary.main' : 'divider',
                      '&:hover': { elevation: 4 }
                    }}
                    onClick={() => handleModelSelection('art', id)}
                  >
                    <CardContent>
                      <Box display="flex" alignItems="center" mb={1}>
                        {selectedArtModel === id ? (
                          <CheckCircle color="primary" sx={{ mr: 1 }} />
                        ) : (
                          <RadioButtonUnchecked sx={{ mr: 1 }} />
                        )}
                        <Typography variant="subtitle1" fontWeight="bold">
                          {model.name}
                        </Typography>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {model.description || 'Traditional art style transformation'}
                      </Typography>
                      
                      <Box display="flex" flexWrap="wrap" gap={0.5} mb={1}>
                        <Chip label={model.type} size="small" color="secondary" />
                        <Chip 
                          label={modelStatus.art?.loaded ? "Ready" : "Load on-demand"} 
                          size="small" 
                          color={modelStatus.art?.loaded ? "success" : "info"}
                        />
                      </Box>
                      
                      <Typography variant="caption" color="text.secondary">
                        Device: {modelStatus.art?.device || 'auto-detect'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Statue Restoration Models */}
      {currentModule === 1 && (
        <Card elevation={3}>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <Restore sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h6">Statue Restoration</Typography>
              <Chip 
                label={modelStatus.statue?.lazy_loading ? "Lazy Loading" : "Standard Loading"} 
                color="primary" 
                size="small" 
                sx={{ ml: 2 }}
              />
            </Box>

            <Grid container spacing={2}>
              {Object.entries(statueModels).map(([id, model]) => (
                <Grid item xs={12} sm={6} md={4} key={id}>
                  <Card 
                    variant={selectedStatueModel === id ? "outlined" : "elevation"}
                    sx={{ 
                      cursor: 'pointer',
                      border: selectedStatueModel === id ? 2 : 1,
                      borderColor: selectedStatueModel === id ? 'primary.main' : 'divider',
                      '&:hover': { elevation: 4 }
                    }}
                    onClick={() => handleModelSelection('statue', id)}
                  >
                    <CardContent>
                      <Box display="flex" alignItems="center" mb={1}>
                        {selectedStatueModel === id ? (
                          <CheckCircle color="primary" sx={{ mr: 1 }} />
                        ) : (
                          <RadioButtonUnchecked sx={{ mr: 1 }} />
                        )}
                        <Typography variant="subtitle1" fontWeight="bold">
                          {model.name}
                        </Typography>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {model.description}
                      </Typography>
                      
                      <Box display="flex" flexWrap="wrap" gap={0.5} mb={1}>
                        <Chip label={model.type} size="small" color="secondary" />
                        <Chip 
                          label={model.weights_exist ? "Ready" : "Weights Missing"} 
                          size="small" 
                          color={model.weights_exist ? "success" : "error"}
                        />
                        <Chip 
                          label={modelStatus.statue?.loaded ? "Loaded" : "Load on-demand"} 
                          size="small" 
                          color={modelStatus.statue?.loaded ? "success" : "info"}
                        />
                      </Box>
                      
                      <Typography variant="caption" color="text.secondary">
                        Device: {model.device || 'auto-detect'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Performance Information */}
      <Card elevation={1} sx={{ mt: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" mb={1}>
            <Speed sx={{ mr: 1, color: 'success.main' }} />
            <Typography variant="subtitle1" color="success.main">
              Memory Optimized Performance
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary">
            ⚡ <strong>Lazy Loading:</strong> Models load only when transformations start<br/>
            🧠 <strong>Memory Efficient:</strong> Automatic GPU cache management<br/>
            🔄 <strong>On-Demand:</strong> No unnecessary memory usage during startup<br/>
            ✨ <strong>Smart Loading:</strong> Automatic device optimization (CUDA/MPS/CPU)
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ModelSelection;
