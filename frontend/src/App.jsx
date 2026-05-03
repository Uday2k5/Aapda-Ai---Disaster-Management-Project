import { useEffect, useMemo, useState } from 'react'
import { Circle, CircleMarker, MapContainer, TileLayer, useMap } from 'react-leaflet'
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  Compass,
  Droplets,
  Gauge,
  ListFilter,
  LocateFixed,
  RefreshCw,
  ShieldCheck,
  SunMoon,
  Waves,
  Zap,
} from 'lucide-react'
import 'leaflet/dist/leaflet.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

const defaultLocation = {
  latitude: 26.1445,
  longitude: 91.7362,
  depth: 18,
}

function App() {
  const [page, setPage] = useState('home')
  const [location, setLocation] = useState(defaultLocation)
  const [flood, setFlood] = useState(null)
  const [earthquake, setEarthquake] = useState(null)
  const [hotspots, setHotspots] = useState({ flood: [], earthquake: [] })
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('Ready')
  const [theme, setTheme] = useState('dark')

  const activeResult = page === 'flood' ? flood : earthquake
  useEffect(() => {
    loadHotspots()
  }, [])

  async function loadHotspots() {
    try {
      const response = await fetch(`${API_BASE}/api/hotspots`)
      if (!response.ok) throw new Error('Hotspot API failed')
      setHotspots(await response.json())
    } catch {
      setHotspots({
        flood: fallbackHotspots.flood,
        earthquake: fallbackHotspots.earthquake,
      })
    }
  }

  async function openHotspot(city) {
    const next = {
      latitude: city.latitude,
      longitude: city.longitude,
      depth: city.depth ?? location.depth,
    }
    setLocation(next)
    await openAnalysis(city.disaster, next)
  }

  async function useLiveLocation(nextPage = page) {
    if (!navigator.geolocation) {
      setMessage('Geolocation is not available in this browser.')
      return
    }

    setLoading(true)
    setMessage('Requesting live location...')
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const next = {
          ...location,
          latitude: Number(position.coords.latitude.toFixed(5)),
          longitude: Number(position.coords.longitude.toFixed(5)),
        }
        setLocation(next)
        setLoading(false)
        setMessage('Live location captured')
        if (nextPage === 'flood' || nextPage === 'earthquake') {
          openAnalysis(nextPage, next)
        }
      },
      () => {
        setLoading(false)
        setMessage('Location permission denied. You can enter coordinates manually.')
      },
      { enableHighAccuracy: true, timeout: 12000 },
    )
  }

  async function openAnalysis(kind, nextLocation = location) {
    setPage(kind)
    await analyze(kind, nextLocation)
  }

  async function analyze(kind = page, nextLocation = location) {
    if (kind !== 'flood' && kind !== 'earthquake') return
    setLoading(true)
    setMessage(`Running ${kind} analysis...`)

    try {
      const endpoint = kind === 'flood' ? '/api/flood/location' : '/api/earthquake/predict'
      const body =
        kind === 'flood'
          ? { latitude: Number(nextLocation.latitude), longitude: Number(nextLocation.longitude) }
          : {
              latitude: Number(nextLocation.latitude),
              longitude: Number(nextLocation.longitude),
              depth: Number(nextLocation.depth),
            }

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!response.ok) throw new Error('API request failed. Make sure FastAPI is running.')
      const data = await response.json()
      if (kind === 'flood') setFlood(data)
      if (kind === 'earthquake') setEarthquake(data)
      setMessage(`${titleFor(kind)} analysis updated`)
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className={`app-shell ${theme}`}>
      <Header
        page={page}
        onNavigate={setPage}
        message={message}
        hotspots={hotspots}
        onSelectHotspot={openHotspot}
        theme={theme}
        onToggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      />

      {page === 'home' && (
        <HomePage
          location={location}
          setLocation={setLocation}
          loading={loading}
          onUseLocation={() => useLiveLocation('home')}
          onOpenFlood={() => openAnalysis('flood')}
          onOpenEarthquake={() => openAnalysis('earthquake')}
          onLiveFlood={() => useLiveLocation('flood')}
          onLiveEarthquake={() => useLiveLocation('earthquake')}
        />
      )}

      {(page === 'flood' || page === 'earthquake') && (
        <AnalysisPage
          kind={page}
          location={location}
          setLocation={setLocation}
          result={activeResult}
          loading={loading}
          onBack={() => setPage('home')}
          onAnalyze={() => analyze(page)}
          onUseLocation={() => useLiveLocation(page)}
        />
      )}

      <Footer />
    </main>
  )
}

function Header({ page, onNavigate, message, hotspots, onSelectHotspot, theme, onToggleTheme }) {
  return (
    <header className="site-header">
      <button className="brand-button" type="button" onClick={() => onNavigate('home')}>
        <ShieldCheck size={24} />
        <span>Disaster IQ</span>
      </button>
      <HotspotPanel hotspots={hotspots} onSelect={onSelectHotspot} />
      <nav>
        <button className={page === 'home' ? 'active' : ''} type="button" onClick={() => onNavigate('home')}>
          Overview
        </button>
        <button className={page === 'flood' ? 'active' : ''} type="button" onClick={() => onNavigate('flood')}>
          Flood
        </button>
        <button className={page === 'earthquake' ? 'active' : ''} type="button" onClick={() => onNavigate('earthquake')}>
          Earthquake
        </button>
        <button type="button" disabled>
          More soon
        </button>
      </nav>
      <button className="theme-toggle" type="button" onClick={onToggleTheme}>
        <SunMoon size={17} />
        {theme === 'dark' ? 'Light' : 'Dark'}
      </button>
      <div className="system-status">
        <Activity size={16} />
        <span>{message}</span>
      </div>
    </header>
  )
}

function HotspotPanel({ hotspots, onSelect }) {
  return (
    <div className="hotspot-shell">
      <button className="hotspot-tab" type="button" aria-label="Show risk cities">
        <ListFilter size={20} />
        <span>Risk Cities</span>
      </button>
      <section className="hotspot-panel">
        <div>
          <p className="eyebrow">Model city watchlist</p>
          <h2>High-risk cities</h2>
        </div>
        <HotspotGroup title="Flood" cities={hotspots.flood} onSelect={onSelect} />
        <HotspotGroup title="Earthquake" cities={hotspots.earthquake} onSelect={onSelect} />
      </section>
    </div>
  )
}

function HotspotGroup({ title, cities, onSelect }) {
  const visible = cities.slice(0, 6)
  return (
    <div className="hotspot-group">
      <h3>{title}</h3>
      {visible.map((city) => (
        <button className="city-row" key={`${city.disaster}-${city.name}`} type="button" onClick={() => onSelect(city)}>
          <span>
            <strong>{city.name}</strong>
            <small>{city.country}</small>
          </span>
          <span className={`city-risk ${riskClass(city.risk_level)}`}>
            {city.risk_level}
          </span>
        </button>
      ))}
    </div>
  )
}

function HomePage({
  location,
  setLocation,
  loading,
  onUseLocation,
  onOpenFlood,
  onOpenEarthquake,
  onLiveFlood,
  onLiveEarthquake,
}) {
  return (
    <section className="home-layout">
      <div className="hero-copy">
        <p className="eyebrow">India live hazard dashboard</p>
        <h1>Choose a disaster model and analyze your current region.</h1>
        <p className="hero-text">
          Capture your live location or enter coordinates manually, then open flood or earthquake analysis with a map-based risk view.
        </p>
        <div className="hero-actions">
          <button type="button" onClick={onUseLocation} disabled={loading}>
            <LocateFixed size={18} />
            Use My Location
          </button>
          <button className="secondary" type="button" onClick={onOpenFlood} disabled={loading}>
            <Waves size={18} />
            Flood Analysis
          </button>
          <button className="secondary" type="button" onClick={onOpenEarthquake} disabled={loading}>
            <AlertTriangle size={18} />
            Earthquake Analysis
          </button>
        </div>
        <section className="signal-strip">
          <div>
            <strong>Live GPS</strong>
            <span>Browser location supported</span>
          </div>
          <div>
            <strong>2 Models</strong>
            <span>Flood and earthquake</span>
          </div>
          <div>
            <strong>Map Risk</strong>
            <span>Low, moderate, high zones</span>
          </div>
        </section>
      </div>

      <LocationPanel location={location} setLocation={setLocation} />

      <section className="choice-grid">
        <ChoiceCard
          icon={<Droplets size={30} />}
          title="Flood Situation"
          description="Live rainfall, India regional flood-proneness, and trained Sentinel-1 model status."
          onOpen={onOpenFlood}
          onLive={onLiveFlood}
          loading={loading}
        />
        <ChoiceCard
          icon={<Zap size={30} />}
          title="Earthquake Prediction"
          description="LSTM magnitude estimate using your pasted earthquake model and recent event sequence."
          onOpen={onOpenEarthquake}
          onLive={onLiveEarthquake}
          loading={loading}
        />
      </section>
    </section>
  )
}

function AnalysisPage({ kind, location, setLocation, result, loading, onBack, onAnalyze, onUseLocation }) {
  const title = titleFor(kind)
  const icon = kind === 'flood' ? <Waves size={26} /> : <AlertTriangle size={26} />
  const metric = kind === 'flood' ? `${result?.risk_percent ?? '--'}%` : result ? `M ${result.predicted_magnitude}` : '--'
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
        </section>
      </div>

      <section className="map-and-info">
        <div className="map-card">
          <RiskMap location={location} level={level} />
          <div className="map-caption">
            <span>{Number(location.latitude).toFixed(4)}, {Number(location.longitude).toFixed(4)}</span>
            <strong>{level} risk zone</strong>
          </div>
        </div>

        <InfoPanel kind={kind} result={result} />
      </section>
    </section>
  )
}

function LocationPanel({ location, setLocation, compact = false }) {
  return (
    <section className={compact ? 'location-card compact' : 'location-card'}>
      <div className="panel-heading">
        <Compass size={22} />
        <div>
          <h2>Selected Location</h2>
          <p>Use GPS or adjust manually.</p>
        </div>
      </div>
      <label>
        Latitude
        <input
          type="number"
          value={location.latitude}
          step="0.0001"
          onChange={(event) => setLocation({ ...location, latitude: event.target.value })}
        />
      </label>
      <label>
        Longitude
        <input
          type="number"
          value={location.longitude}
          step="0.0001"
          onChange={(event) => setLocation({ ...location, longitude: event.target.value })}
        />
      </label>
      <label>
        Earthquake depth estimate
        <input
          type="number"
          value={location.depth}
          min="0"
          max="700"
          onChange={(event) => setLocation({ ...location, depth: event.target.value })}
        />
      </label>
    </section>
  )
}

function ChoiceCard({ icon, title, description, onOpen, onLive, loading }) {
  return (
    <article className="choice-card">
      <div className="choice-icon">{icon}</div>
      <h2>{title}</h2>
      <p>{description}</p>
      <div className="choice-actions">
        <button type="button" onClick={onLive} disabled={loading}>
          <LocateFixed size={18} />
          Live + Open
        </button>
        <button className="secondary" type="button" onClick={onOpen} disabled={loading}>
          Analyze
        </button>
      </div>
    </article>
  )
}

function InfoPanel({ kind, result }) {
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
    </section>
  )
}

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

function Footer() {
  return (
    <footer className="site-footer">
      <span>Disaster IQ for Minor Project 2</span>
      <span>Flood and earthquake predictions are academic decision-support outputs, not official alerts.</span>
    </footer>
  )
}

function RiskMap({ location, level }) {
  const latitude = Number(location.latitude)
  const longitude = Number(location.longitude)
  const center = useMemo(() => [latitude, longitude], [latitude, longitude])
  const className = riskClass(level)
  const color = riskColor(className)

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
    </MapContainer>
  )
}

function RecenterMap({ center }) {
  const map = useMap()
  useEffect(() => {
    map.setView(center, map.getZoom(), { animate: true })
  }, [center, map])
  return null
}

function titleFor(kind) {
  return kind === 'flood' ? 'Flood Analysis' : 'Earthquake Analysis'
}

function riskClass(level) {
  const value = String(level).toLowerCase()
  if (value.includes('high')) return 'high'
  if (value.includes('moderate')) return 'moderate'
  if (value.includes('low')) return 'low'
  return 'waiting'
}

function riskColor(level) {
  if (level === 'high') return '#ff5858'
  if (level === 'moderate') return '#ffcd47'
  if (level === 'low') return '#40e881'
  return '#d2e0dd'
}

const fallbackHotspots = {
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
}

export default App
