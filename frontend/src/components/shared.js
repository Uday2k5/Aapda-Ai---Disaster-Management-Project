export function titleFor(kind) {
  if (kind === 'flood') return 'Flood Analysis'
  if (kind === 'earthquake') return 'Earthquake Analysis'
  return 'Wildfire Analysis'
}

export function riskClass(level) {
  const value = String(level).toLowerCase()
  if (value.includes('high')) return 'high'
  if (value.includes('moderate')) return 'moderate'
  if (value.includes('low')) return 'low'
  return 'waiting'
}

export function riskColor(level) {
  if (level === 'high') return '#ff5858'
  if (level === 'moderate') return '#ffcd47'
  if (level === 'low') return '#40e881'
  return '#d2e0dd'
}

export const fallbackHotspots = {
  flood: [
    { name: 'Guwahati', country: 'India', disaster: 'flood', latitude: 26.1445, longitude: 91.7362, depth: 18, risk_level: 'High', risk_score: 0.82, risk_percent: 82 },
    { name: 'Patna', country: 'India', disaster: 'flood', latitude: 25.5941, longitude: 85.1376, depth: 18, risk_level: 'High', risk_score: 0.78, risk_percent: 78 },
    { name: 'Kolkata', country: 'India', disaster: 'flood', latitude: 22.5726, longitude: 88.3639, depth: 18, risk_level: 'Moderate', risk_score: 0.61, risk_percent: 61 },
  ],
  earthquake: [
    { name: 'Srinagar', country: 'India', disaster: 'earthquake', latitude: 34.0837, longitude: 74.7973, depth: 15, risk_level: 'High', risk_score: 0.78, risk_percent: 78 },
    { name: 'Gangtok', country: 'India', disaster: 'earthquake', latitude: 27.3314, longitude: 88.6138, depth: 18, risk_level: 'High', risk_score: 0.75, risk_percent: 75 },
    { name: 'Kathmandu', country: 'Nepal', disaster: 'earthquake', latitude: 27.7172, longitude: 85.324, depth: 12, risk_level: 'High', risk_score: 0.82, risk_percent: 82 },
  ],
  wildfire: [
    { name: 'Dehradun', country: 'India', disaster: 'wildfire', latitude: 30.3165, longitude: 78.0322, depth: 18, risk_level: 'Low', risk_score: 0.22, risk_percent: 22 },
    { name: 'Shimla', country: 'India', disaster: 'wildfire', latitude: 31.1048, longitude: 77.1734, depth: 18, risk_level: 'Low', risk_score: 0.21, risk_percent: 21 },
    { name: 'Srinagar', country: 'India', disaster: 'wildfire', latitude: 34.0837, longitude: 74.7973, depth: 18, risk_level: 'Low', risk_score: 0.17, risk_percent: 17 },
  ],
}
