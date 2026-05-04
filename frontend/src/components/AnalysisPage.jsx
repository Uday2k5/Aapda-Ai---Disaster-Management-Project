import { useEffect, useMemo } from 'react'
import { Circle, CircleMarker, MapContainer, Polyline, TileLayer, useMap } from 'react-leaflet'
import { AlertTriangle, ArrowLeft, BarChart3, Flame, Gauge, LocateFixed, RefreshCw, Waves } from 'lucide-react'

import { LocationPanel } from './HomePage'
import { riskClass, riskColor, titleFor } from './shared'

function InfoHeader({ title, subtitle }) {
  return (
    <div className="panel-heading">
      <Gauge size={22} />
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  )
}

function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function StepList({ route }) {
  if (!route?.needed || !route.steps?.length) return null
  return (
    <div className="step-list">
      <h2>Turn guidance</h2>
      {route.steps.map((step, index) => (
        <div className="step-row" key={`${step.instruction}-${index}`}>
          <span>{index + 1}. {step.instruction}</span>
          <strong>{step.distance_m} m</strong>
        </div>
      ))}
    </div>
  )
}

function InfoPanel({ kind, result, route }) {
  if (!result) {
    return (
      <section className="info-panel">
        <div className="panel-heading">
          <BarChart3 size={22} />
          <div>
            <h2>Analysis Details</h2>
            <p>Run analysis to load model output.</p>
          </div>
        </div>
      </section>
    )
  }

  if (kind === 'flood') {
    return (
      <section className="info-panel">
        <InfoHeader title="Flood Intelligence" subtitle={result.advice} />
        <InfoRow label="Risk percent" value={`${result.risk_percent}%`} />
        <InfoRow label="Rain now" value={`${result.weather.rain_now_mm} mm`} />
        <InfoRow label="24 hour rain" value={`${result.weather.rain_24h_mm} mm`} />
        <InfoRow label="72 hour rain" value={`${result.weather.rain_72h_mm} mm`} />
        <InfoRow label="Sensitive region" value={result.nearest_sensitive_region} />
        <InfoRow label="Sentinel checkpoint" value={result.model_status.checkpoint_exists ? 'Available' : 'Missing'} />
        <InfoRow label="Flood model IoU" value={result.model_status.trained_best_iou?.toFixed(3) ?? '--'} />
        <InfoRow label="Safe path" value={route?.needed ? route.end_risk_level : 'Not needed'} />
        <StepList route={route} />
      </section>
    )
  }

  if (kind === 'wildfire') {
    return (
      <section className="info-panel">
        <InfoHeader title="Wildfire Intelligence" subtitle={result.advice} />
        <InfoRow label="Risk percent" value={`${result.risk_percent}%`} />
        <InfoRow label="Grid cell" value={`${result.lat_bin}, ${result.lon_bin}`} />
        <InfoRow label="Recent fire days" value={result.recent_fire_days} />
        <InfoRow label="Mean FRP" value={result.mean_frp} />
        <InfoRow label="Seasonal factor" value={result.seasonal_factor} />
        <InfoRow label="Sensitive region" value={result.nearest_fire_region} />
        <InfoRow label="Model status" value={String(result.model_status).startsWith('loaded') ? 'Loaded' : 'Fallback'} />
        <InfoRow label="Safe path" value={route?.needed ? route.end_risk_level : 'Not needed'} />
        <StepList route={route} />
      </section>
    )
  }

  return (
    <section className="info-panel">
      <InfoHeader title="Earthquake Intelligence" subtitle={result.note} />
      <InfoRow label="Predicted magnitude" value={`M ${result.predicted_magnitude}`} />
      <InfoRow label="Risk percent" value={`${result.risk_percent}%`} />
      <InfoRow label="Depth" value={`${result.depth_km} km`} />
      <InfoRow label="Seismic zone" value={result.seismic_zone ?? 'General background'} />
      <InfoRow label="Model status" value={result.model_status === 'loaded' ? 'Loaded' : 'Fallback'} />
      <InfoRow label="Historical events" value={result.data_points} />
      <InfoRow label="Safe path" value={route?.needed ? route.end_risk_level : 'Not needed'} />
      <StepList route={route} />
    </section>
  )
}

function RecenterMap({ center }) {
  const map = useMap()
  useEffect(() => {
    map.setView(center, map.getZoom(), { animate: true })
  }, [center, map])
  return null
}

function RiskMap({ location, level, route }) {
  const latitude = Number(location.latitude)
  const longitude = Number(location.longitude)
  const center = useMemo(() => [latitude, longitude], [latitude, longitude])
  const color = riskColor(riskClass(level))

  return (
    <MapContainer center={center} zoom={11} minZoom={4} maxZoom={18} scrollWheelZoom className="leaflet-map">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <RecenterMap center={center} />
      <Circle
        center={center}
        radius={8500}
        pathOptions={{
          color,
          fillColor: color,
          fillOpacity: 0.26,
          weight: 3,
        }}
      />
      <CircleMarker
        center={center}
        radius={8}
        pathOptions={{
          color: '#ffffff',
          fillColor: color,
          fillOpacity: 1,
          weight: 3,
        }}
      />
      {route?.needed && route.path?.length > 1 && (
        <Polyline
          positions={route.path}
          pathOptions={{
            color: '#4da3ff',
            weight: 5,
            opacity: 0.9,
          }}
        />
      )}
    </MapContainer>
  )
}

export function AnalysisPage({ kind, location, setLocation, result, route, loading, onBack, onAnalyze, onUseLocation }) {
  const title = titleFor(kind)
  const icon = kind === 'flood' ? <Waves size={26} /> : kind === 'earthquake' ? <AlertTriangle size={26} /> : <Flame size={26} />
  const metric =
    kind === 'flood' ? `${result?.risk_percent ?? '--'}%` : kind === 'earthquake' ? (result ? `M ${result.predicted_magnitude}` : '--') : `${result?.risk_percent ?? '--'}%`
  const level = result?.risk_level ?? 'Waiting'

  return (
    <section className="analysis-layout">
      <div className="page-heading">
        <button className="ghost-button" type="button" onClick={onBack}>
          <ArrowLeft size={18} />
          Back
        </button>
        <div>
          <p className="eyebrow">{title}</p>
          <h1>{title} map for your selected region.</h1>
        </div>
      </div>

      <div className="analysis-grid">
        <LocationPanel location={location} setLocation={setLocation} compact />

        <section className={`risk-score ${riskClass(level)}`}>
          <div className="risk-title">
            {icon}
            <span>{title}</span>
          </div>
          <strong>{metric}</strong>
          <span className="level-pill">{level} risk</span>
          <div className="button-stack">
            <button type="button" onClick={onAnalyze} disabled={loading}>
              <RefreshCw size={18} />
              Re-analyze
            </button>
            <button className="secondary" type="button" onClick={onUseLocation} disabled={loading}>
              <LocateFixed size={18} />
              Live location
            </button>
          </div>
          {route?.needed && (
            <p className="route-note">
              Safest exit route available. End zone: {route.end_risk_level} ({Math.round(route.end_risk_score * 100)}%). Mode: {route.route_mode === 'road' ? 'roads' : 'risk fallback'}.
            </p>
          )}
        </section>
      </div>

      <section className="map-and-info">
        <div className="map-card">
          <RiskMap location={location} level={level} route={route} />
          <div className="map-caption">
            <span>{Number(location.latitude).toFixed(4)}, {Number(location.longitude).toFixed(4)}</span>
            <strong>{level} risk zone</strong>
          </div>
        </div>

        <InfoPanel kind={kind} result={result} route={route} />
      </section>
    </section>
  )
}
