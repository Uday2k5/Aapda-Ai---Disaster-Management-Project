import { Activity, ListFilter, ShieldCheck, SunMoon } from 'lucide-react'

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
        <HotspotGroup title="Wildfire" cities={hotspots.wildfire} onSelect={onSelect} />
      </section>
    </div>
  )
}

export function Header({ page, onNavigate, message, hotspots, onSelectHotspot, theme, onToggleTheme }) {
  return (
    <header className="site-header">
      <button className="brand-button" type="button" onClick={() => onNavigate('home')}>
        <ShieldCheck size={24} />
        <span>Aapda Ai</span>
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
        <button className={page === 'wildfire' ? 'active' : ''} type="button" onClick={() => onNavigate('wildfire')}>
          Wildfire
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

export function Footer() {
  return (
    <footer className="site-footer">
      <span>Aapda Ai</span>
    </footer>
  )
}
