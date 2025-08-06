import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  CircularProgress,
  Paper,
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Divider
} from '@mui/material';
import {
  CloudUpload,
  AutoFixHigh,
  Download,
  Refresh,
  Settings,
  Tune,
  ExpandMore,
  RestoreFromTrash,
  PhotoCamera,
  Architecture
} from '@mui/icons-material';
import ImageUpload from './ImageUpload';
import ImagePreview from './ImagePreview';

const StatueRestoration = () => {
  // State management
  const [originalImage, setOriginalImage] = useState(null);
  const [maskImage, setMaskImage] = useState(null);
  const [restoredImage, setRestoredImage] = useState(null);
  const [comparisonImage, setComparisonImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentTab, setCurrentTab] = useState(0);
  
  // Restoration parameters
  const [prompt, setPrompt] = useState('beautiful classical marble statue with delicate carved features, weathered ancient stone texture, renaissance sculpture, detailed craftsmanship');
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [numSteps, setNumSteps] = useState(50);
  const [strength, setStrength] = useState(0.8);
  const [selectedPreset, setSelectedPreset] = useState('classical_marble');
  
  // Advanced parameters
  const [seedValue, setSeedValue] = useState(-1);
  const [enableFaceEnhancement, setEnableFaceEnhancement] = useState(true);
  const [preserveTexture, setPreserveTexture] = useState(true);
  const [enhanceDetails, setEnhanceDetails] = useState(false);
  const [colorCorrection, setColorCorrection] = useState(true);
  
  // Mask generation parameters
  const [edgeThreshold1, setEdgeThreshold1] = useState(50);
  const [edgeThreshold2, setEdgeThreshold2] = useState(150);
  const [dilateIterations, setDilateIterations] = useState(2);
  const [kernelSize, setKernelSize] = useState(5);
  const [autoGenerateMask, setAutoGenerateMask] = useState(true);
  
  // Module status
  const [moduleStatus, setModuleStatus] = useState(null);

  // Predefined restoration presets
  const restorationPresets = {
    classical_marble: {
      name: 'Classical Marble',
      description: 'Ancient Greek/Roman marble sculptures',
      prompt: 'beautiful classical marble statue with delicate carved features, weathered ancient stone texture, renaissance sculpture, detailed craftsmanship',
      guidance_scale: 7.5,
      num_inference_steps: 50,
      strength: 0.8
    },
    renaissance_sculpture: {
      name: 'Renaissance Sculpture',
      description: 'Detailed Renaissance-era sculptures',
      prompt: 'masterpiece renaissance marble sculpture, michelangelo style, perfect proportions, detailed carved features, classical beauty',
      guidance_scale: 8.0,
      num_inference_steps: 60,
      strength: 0.75
    },
    ancient_bronze: {
      name: 'Ancient Bronze',
      description: 'Weathered bronze statues',
      prompt: 'ancient bronze statue with patina, weathered metal surface, classical figure, archaeological artifact',
      guidance_scale: 7.0,
      num_inference_steps: 45,
      strength: 0.85
    },
    modern_sculpture: {
      name: 'Modern Sculpture',
      description: 'Contemporary artistic sculptures',
      prompt: 'modern artistic sculpture, clean lines, contemporary style, polished surface, artistic masterpiece',
      guidance_scale: 6.5,
      num_inference_steps: 40,
      strength: 0.7
    },
    bust_portrait: {
      name: 'Portrait Bust',
      description: 'Classical portrait busts',
      prompt: 'classical portrait bust, detailed facial features, noble expression, marble sculpture, museum quality',
      guidance_scale: 8.5,
      num_inference_steps: 55,
      strength: 0.8
    }
  };

  // Load module status on component mount
  useEffect(() => {
    checkModuleStatus();
  }, []);

  const checkModuleStatus = async () => {
    try {
      const response = await fetch('http://localhost:5002/api/statue-restoration/health');
      const data = await response.json();
      setModuleStatus(data);
    } catch (error) {
      console.error('Failed to check module status:', error);
      setModuleStatus({ status: 'error', error: 'Service unavailable' });
    }
  };

  const applyPreset = (presetKey) => {
    const preset = restorationPresets[presetKey];
    if (preset) {
      setPrompt(preset.prompt);
      setGuidanceScale(preset.guidance_scale);
      setNumSteps(preset.num_inference_steps);
      setStrength(preset.strength);
      setSelectedPreset(presetKey);
    }
  };

  const handleImageUpload = (imageData) => {
    setOriginalImage(imageData);
    setMaskImage(null);
    setRestoredImage(null);
    setComparisonImage(null);
    setError('');
  };

  const generateMask = async () => {
    if (!originalImage) {
      setError('Please upload an image first');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5002/api/statue-restoration/generate-mask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: originalImage,
          edge_threshold1: edgeThreshold1,
          edge_threshold2: edgeThreshold2,
          dilate_iterations: dilateIterations,
          kernel_size: kernelSize
        }),
      });

      const data = await response.json();

      if (data.success) {
        setMaskImage(data.mask);
      } else {
        setError(data.error || 'Failed to generate mask');
      }
    } catch (error) {
      setError('Failed to generate mask: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const performRestoration = async () => {
    if (!originalImage) {
      setError('Please upload an image first');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const requestBody = {
        image: originalImage,
        prompt: prompt,
        guidance_scale: guidanceScale,
        num_inference_steps: numSteps,
        strength: strength,
        seed: seedValue,
        enable_face_enhancement: enableFaceEnhancement,
        preserve_texture: preserveTexture,
        enhance_details: enhanceDetails,
        color_correction: colorCorrection
      };

      // Include mask if not auto-generating or if manually created
      if (!autoGenerateMask && maskImage) {
        requestBody.mask = maskImage;
      }

      const response = await fetch('http://localhost:5002/api/statue-restoration/restore', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (data.success) {
        setRestoredImage(data.restored_image);
        setMaskImage(data.mask_used);
        setComparisonImage(data.comparison);
      } else {
        setError(data.error || 'Failed to restore statue');
      }
    } catch (error) {
      setError('Failed to restore statue: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = (imageData, filename) => {
    if (!imageData) return;

    const link = document.createElement('a');
    link.href = imageData;
    link.download = `statue_restoration_${filename}_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleReset = () => {
    setOriginalImage(null);
    setMaskImage(null);
    setRestoredImage(null);
    setComparisonImage(null);
    setError('');
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          🏛️ Statue Restoration
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          AI-powered restoration of damaged classical statues and sculptures
        </Typography>
      </Box>

      {/* Module Status */}
      {moduleStatus && !moduleStatus.status?.is_loaded && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Restoration pipeline is not loaded. It will be initialized automatically on first use.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Left Panel - Controls */}
        <Grid item xs={12} md={4}>
          <Card elevation={3}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Settings sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h6">Restoration Settings</Typography>
              </Box>

              {/* Preset Selection */}
              <FormControl fullWidth margin="normal">
                <InputLabel>Restoration Preset</InputLabel>
                <Select
                  value={selectedPreset}
                  onChange={(e) => applyPreset(e.target.value)}
                  label="Restoration Preset"
                >
                  {Object.entries(restorationPresets).map(([key, preset]) => (
                    <MenuItem key={key} value={key}>
                      <Box>
                        <Typography variant="body2" fontWeight="medium">
                          {preset.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" display="block">
                          {preset.description}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Restoration Prompt */}
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Restoration Description"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                margin="normal"
                variant="outlined"
                helperText="Describe the desired restoration style and details"
              />

              {/* Advanced Settings */}
              <Box sx={{ mt: 2, mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Advanced Parameters
                </Typography>

                {/* Guidance Scale */}
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Guidance Scale: {guidanceScale}
                  </Typography>
                  <Slider
                    value={guidanceScale}
                    onChange={(e, newValue) => setGuidanceScale(newValue)}
                    min={1.0}
                    max={20.0}
                    step={0.5}
                    marks={[
                      { value: 1, label: '1' },
                      { value: 7.5, label: '7.5' },
                      { value: 15, label: '15' }
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Box>

                {/* Strength */}
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Restoration Strength: {strength}
                  </Typography>
                  <Slider
                    value={strength}
                    onChange={(e, newValue) => setStrength(newValue)}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    marks={[
                      { value: 0.1, label: '0.1' },
                      { value: 0.8, label: '0.8' },
                      { value: 1.0, label: '1.0' }
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Box>

                {/* Inference Steps */}
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Inference Steps: {numSteps}
                  </Typography>
                  <Slider
                    value={numSteps}
                    onChange={(e, newValue) => setNumSteps(newValue)}
                    min={20}
                    max={100}
                    step={5}
                    marks={[
                      { value: 20, label: '20' },
                      { value: 50, label: '50' },
                      { value: 100, label: '100' }
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Box>
              </Box>

              {/* Mask Settings */}
              <Box sx={{ mt: 2, mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Damage Detection
                </Typography>
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={autoGenerateMask}
                      onChange={(e) => setAutoGenerateMask(e.target.checked)}
                    />
                  }
                  label="Auto-generate damage mask"
                />

                {!autoGenerateMask && (
                  <Button
                    variant="outlined"
                    onClick={generateMask}
                    disabled={isLoading || !originalImage}
                    startIcon={<Tune />}
                    fullWidth
                    sx={{ mt: 1 }}
                  >
                    Generate Mask
                  </Button>
                )}
              </Box>

              {/* Action Buttons */}
              <Box mt={3} display="flex" flexDirection="column" gap={1}>
                <Button
                  variant="contained"
                  onClick={performRestoration}
                  disabled={isLoading || !originalImage}
                  startIcon={isLoading ? <CircularProgress size={20} /> : <AutoFixHigh />}
                  fullWidth
                  sx={{ 
                    background: 'linear-gradient(45deg, #8B4513 30%, #D2691E 90%)',
                    py: 1.5
                  }}
                >
                  {isLoading ? 'Restoring Statue...' : 'Restore Statue'}
                </Button>

                <Box display="flex" gap={1}>
                  <Button
                    variant="outlined"
                    onClick={handleReset}
                    startIcon={<Refresh />}
                    fullWidth
                  >
                    Reset
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => handleDownload(restoredImage, 'restored')}
                    disabled={!restoredImage}
                    startIcon={<Download />}
                    fullWidth
                  >
                    Download
                  </Button>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Right Panel - Images */}
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Tabs 
              value={currentTab} 
              onChange={(e, newValue) => setCurrentTab(newValue)}
              sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
            >
              <Tab label="Upload & Preview" />
              <Tab label="Restoration Result" />
              <Tab label="Comparison View" />
            </Tabs>

            {/* Tab 0: Upload & Preview */}
            {currentTab === 0 && (
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Original Image
                  </Typography>
                  <ImageUpload onImageUpload={handleImageUpload} />
                  {originalImage && (
                    <Box mt={2}>
                      <ImagePreview src={originalImage} alt="Original Statue" />
                    </Box>
                  )}
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Damage Mask
                  </Typography>
                  {maskImage ? (
                    <ImagePreview src={maskImage} alt="Damage Mask" />
                  ) : (
                    <Box 
                      display="flex" 
                      alignItems="center" 
                      justifyContent="center" 
                      height={300}
                      bgcolor="grey.100"
                      borderRadius={1}
                    >
                      <Typography variant="body2" color="text.secondary">
                        Damage mask will appear here
                      </Typography>
                    </Box>
                  )}
                </Grid>
              </Grid>
            )}

            {/* Tab 1: Restoration Result */}
            {currentTab === 1 && (
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Before Restoration
                  </Typography>
                  {originalImage ? (
                    <ImagePreview src={originalImage} alt="Before Restoration" />
                  ) : (
                    <Box 
                      display="flex" 
                      alignItems="center" 
                      justifyContent="center" 
                      height={300}
                      bgcolor="grey.100"
                      borderRadius={1}
                    >
                      <Typography variant="body2" color="text.secondary">
                        Original image will appear here
                      </Typography>
                    </Box>
                  )}
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    After Restoration
                  </Typography>
                  {isLoading ? (
                    <Box 
                      display="flex" 
                      flexDirection="column" 
                      alignItems="center" 
                      justifyContent="center" 
                      height={300}
                    >
                      <CircularProgress size={60} />
                      <Typography variant="body2" mt={2} color="text.secondary">
                        AI is restoring your statue...
                      </Typography>
                    </Box>
                  ) : restoredImage ? (
                    <ImagePreview src={restoredImage} alt="Restored Statue" />
                  ) : (
                    <Box 
                      display="flex" 
                      alignItems="center" 
                      justifyContent="center" 
                      height={300}
                      bgcolor="grey.100"
                      borderRadius={1}
                    >
                      <Typography variant="body2" color="text.secondary">
                        Restored statue will appear here
                      </Typography>
                    </Box>
                  )}
                </Grid>
              </Grid>
            )}

            {/* Tab 2: Comparison View */}
            {currentTab === 2 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Side-by-Side Comparison
                </Typography>
                {comparisonImage ? (
                  <ImagePreview src={comparisonImage} alt="Comparison: Original | Mask | Restored" />
                ) : (
                  <Box 
                    display="flex" 
                    alignItems="center" 
                    justifyContent="center" 
                    height={300}
                    bgcolor="grey.100"
                    borderRadius={1}
                  >
                    <Typography variant="body2" color="text.secondary">
                      Comparison view will appear here after restoration
                    </Typography>
                  </Box>
                )}
                {comparisonImage && (
                  <Typography variant="caption" display="block" mt={1} textAlign="center">
                    Left: Original | Center: Damage Mask | Right: Restored
                  </Typography>
                )}
              </Box>
            )}

            {/* Error Display */}
            {error && (
              <Box mt={2}>
                <Alert severity="error" onClose={() => setError('')}>
                  {error}
                </Alert>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default StatueRestoration;
