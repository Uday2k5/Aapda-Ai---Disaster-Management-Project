import { useEffect, useState } from 'react'
import 'leaflet/dist/leaflet.css'

import { AnalysisPage } from './components/AnalysisPage'
import { Header, Footer } from './components/Header'
import { HomePage } from './components/HomePage'
import { fallbackHotspots, titleFor } from './components/shared'

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
  const [wildfire, setWildfire] = useState(null)
  const [routes, setRoutes] = useState({ flood: null, earthquake: null, wildfire: null })
  const [hotspots, setHotspots] = useState({ flood: [], earthquake: [], wildfire: [] })
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('Ready')
  const [theme, setTheme] = useState('light')

  const activeResult = page === 'flood' ? flood : page === 'earthquake' ? earthquake : wildfire
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
        wildfire: fallbackHotspots.wildfire,
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
        if (nextPage === 'flood' || nextPage === 'earthquake' || nextPage === 'wildfire') {
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
    if (kind !== 'flood' && kind !== 'earthquake' && kind !== 'wildfire') return
    setLoading(true)
    setMessage(`Running ${kind} analysis...`)

    try {
      const endpoint =
        kind === 'flood' ? '/api/flood/location' : kind === 'earthquake' ? '/api/earthquake/predict' : '/api/wildfire/predict'
      const body =
        kind === 'flood'
          ? { latitude: Number(nextLocation.latitude), longitude: Number(nextLocation.longitude) }
          : kind === 'earthquake'
            ? {
              latitude: Number(nextLocation.latitude),
              longitude: Number(nextLocation.longitude),
              depth: Number(nextLocation.depth),
            }
            : { latitude: Number(nextLocation.latitude), longitude: Number(nextLocation.longitude) }

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!response.ok) throw new Error('API request failed. Make sure FastAPI is running.')
      const data = await response.json()
      if (kind === 'flood') setFlood(data)
      if (kind === 'earthquake') setEarthquake(data)
      if (kind === 'wildfire') setWildfire(data)
      if (data.risk_level === 'High') {
        await loadSafestRoute(kind, nextLocation)
      } else {
        setRoutes((current) => ({ ...current, [kind]: null }))
      }
      setMessage(`${titleFor(kind)} analysis updated`)
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  async function loadSafestRoute(kind, nextLocation) {
    try {
      const response = await fetch(`${API_BASE}/api/route/safest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          disaster: kind,
          latitude: Number(nextLocation.latitude),
          longitude: Number(nextLocation.longitude),
          depth: Number(nextLocation.depth),
        }),
      })
      if (!response.ok) throw new Error('Route API request failed')
      const data = await response.json()
      setRoutes((current) => ({ ...current, [kind]: data }))
    } catch {
      setRoutes((current) => ({ ...current, [kind]: null }))
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
          onOpenWildfire={() => openAnalysis('wildfire')}
          onLiveFlood={() => useLiveLocation('flood')}
          onLiveEarthquake={() => useLiveLocation('earthquake')}
          onLiveWildfire={() => useLiveLocation('wildfire')}
        />
      )}

      {(page === 'flood' || page === 'earthquake' || page === 'wildfire') && (
        <AnalysisPage
          kind={page}
          location={location}
          setLocation={setLocation}
          result={activeResult}
          route={routes[page]}
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

export default App
