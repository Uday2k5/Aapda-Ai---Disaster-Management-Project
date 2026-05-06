import { AlertTriangle, Compass, Droplets, Flame, LocateFixed, Waves, Zap } from 'lucide-react'

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

export function HomePage({
  location,
  setLocation,
  loading,
  onUseLocation,
  onOpenFlood,
  onOpenEarthquake,
  onOpenWildfire,
  onLiveFlood,
  onLiveEarthquake,
  onLiveWildfire,
}) {
  return (
    <section className="home-layout">
      <div className="hero-copy">
        <p className="eyebrow">India live hazard dashboard</p>
        <h1>Disaster analysis for your region.</h1>
        <p className="hero-text">
          Capture your live location or enter coordinates manually, then open flood, earthquake, or wildfire analysis with a map-based risk view.
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
          <button className="secondary" type="button" onClick={onOpenWildfire} disabled={loading}>
            <Flame size={18} />
            Wildfire Analysis
          </button>
        </div>
        <section className="signal-strip">
          <div>
            <strong>Live GPS</strong>
            <span>Browser location supported</span>
          </div>
          <div>
            <strong>3 Models</strong>
            <span>Flood, earthquake, wildfire</span>
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
        <ChoiceCard
          icon={<Flame size={30} />}
          title="Wildfire Risk"
          description="Historical fire-grid model with regional priors for next-day hotspot risk and escape routing."
          onOpen={onOpenWildfire}
          onLive={onLiveWildfire}
          loading={loading}
        />
      </section>
    </section>
  )
}

export { LocationPanel }
