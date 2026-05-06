import { useEffect, useState } from 'react'
import { Activity, ListFilter, MoonStar, ShieldCheck, Sun } from 'lucide-react'

import { riskClass } from './shared'

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

function HotspotPanel({ hotspots, onSelect }) {
  return (
    <div className="hotspot-shell">
      <button className="nav-pill-button" type="button" aria-label="Show risk cities">
        <ListFilter size={16} />
        <span>Risk Cities</span>
      </button>
      <section className="hotspot-panel">
        <div>
          <p className="eyebrow">Model city watchlist</p>
          <h2>High-risk cities</h2>
        </div>
        <HotspotGroup title="Flood" cities={hotspots.flood} onSelect={onSelect} />
        <HotspotGroup title="Earthquake" cities={hotspots.earthquake} onSelect={onSelect} />
        <HotspotGroup title="Wildfire" cities={hotspots.wildfire} onSelect={onSelect} />
      </section>
    </div>
  )
}

export function Header({ page, onNavigate, message, hotspots, onSelectHotspot, theme, onToggleTheme }) {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    function handleScroll() {
      const trigger = page === 'home' ? window.innerHeight * 0.55 : 8
      setScrolled(window.scrollY > trigger)
    }

    handleScroll()
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [page])

  return (
    <header className={`site-header ${page === 'home' ? 'home-header' : 'inner-header'} ${scrolled ? 'scrolled' : ''}`}>
      <button className="brand-button" type="button" onClick={() => onNavigate('home')}>
        <ShieldCheck size={24} />
        <span>Aapda Ai</span>
      </button>
      <div className="header-center">
        <nav className="nav-pill">
          <button className={page === 'home' ? 'active' : ''} type="button" onClick={() => onNavigate('home')}>
            Overview
          </button>
          <button className={page === 'flood' ? 'active' : ''} type="button" onClick={() => onNavigate('flood')}>
            Flood
          </button>
          <button className={page === 'earthquake' ? 'active' : ''} type="button" onClick={() => onNavigate('earthquake')}>
            Earthquake
          </button>
          <button className={page === 'wildfire' ? 'active' : ''} type="button" onClick={() => onNavigate('wildfire')}>
            Wildfire
          </button>
          <HotspotPanel hotspots={hotspots} onSelect={onSelectHotspot} />
        </nav>
      </div>
      <div className="header-actions">
        <div className="status-orb" title={message} aria-label={message}>
          <Activity size={16} />
        </div>
        <button className="theme-orb" type="button" onClick={onToggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
          {theme === 'dark' ? <Sun size={17} /> : <MoonStar size={17} />}
        </button>
      </div>
    </header>
  )
}

export function Footer() {
  return (
    <footer className="site-footer">
      <span>Aapda Ai</span>
    </footer>
  )
}
