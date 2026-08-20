# Coloft - A Soft Place to Land

A vibrant static website for Coloft, a grassroots healing collective in Arcata, CA (Goudi'ni, Wiyot Land).

## Features

- **Local Events**: Printable flyers for 3 active recurring Humboldt County events (starting mid-January 2026)
- **Regional Community Calendar**: 36 transformational events across 10 regions (SF to Ashland)
- **Mobile Responsive**: Works beautifully on all devices
- **Print Optimized**: All event flyers fit perfectly on 1 page
- **Data-Driven**: Regional calendar built from JSON for easy maintenance

## Quick Start

```bash
# Build regional calendar from data
npm run build:regional

# Run tests
npm install
npm test

# Print local event flyers
Open index.html in browser → Print Preview → Save as PDF

# Deploy to GitHub Pages
Push to GitHub → Settings → Pages → Enable
```

## Project Structure

```
/
├── index.html              # Main landing page (Coloft local events)
├── regional-calendar.html  # Regional community calendar (generated from regional-events.json)
├── regional-events.json    # Regional calendar data (24 events, 10 regions)
├── events.json             # Local event metadata (not currently used by date calculation)
├── events/                 # Individual event pages with printable flyers
├── scripts/                # Maintenance & build scripts
│   ├── build-regional-calendar.js  # Build regional-calendar.html from JSON
│   ├── dates.js            # Browser-side dynamic date calculations
│   ├── verify-order.js     # Validate region ordering
│   ├── validate-links.js   # Check external URLs
│   ├── validate-event-data.js  # Validate regional event data quality
│   └── check-stale-dates.js    # Identify events needing quarterly updates
├── tests/                  # Automated tests
│   ├── test-all-print.js   # Verify 1-page print constraint
│   └── test-dates.js       # Verify date calculations
└── .claude/                # AI assistant context
    ├── instructions.md     # Project guidelines for Claude
    └── settings.local.json # Permissions
```

## Testing & Validation

```bash
npm run validate-data       # Validate event data quality (run before building)
npm run build:regional      # Build regional-calendar.html from JSON
npm test                    # Run all tests (print + dates)
npm run test:print          # Test print constraint only
npm run test:dates          # Test date calculations only
npm run verify-calendar     # Validate region ordering
npm run validate-links      # Check all external links (may take ~1 min)
npm run check-stale-dates   # Identify events needing quarterly date updates
```

## Design Principles

### Color Palette
- **Purple** `#7C3AED` - Primary
- **Green** `#059669` - Secondary
- **Cyan** `#0891B2` - Accent
- **Terracotta** `#DC6B4A` - Warm accent
- **Pink** `#E91E63` - Highlight
- **Amber** `#F4B860` - Glow

### Typography & Theme
- "Co-" prefix throughout: co-llective, co-nnection, co-mmunity
- Land acknowledgment: Wiyot land, Goudi'ni
- Visual hierarchy: Icons first, text secondary

### Print Constraint
**Critical**: All event flyers MUST fit on 1 page (US Letter, portrait)
- Tested automatically via `npm run test:print`
- No browser headers/footers (`@page margin: 0`)

## Regional Community Calendar

The regional community calendar features transformational events focused on:
- **Movement**: Ecstatic dance, 5Rhythms, Contact Improvisation, Tai Chi, Qigong
- **Healing**: Breathwork, sound healing, somatic practices, embodiment work
- **Connection**: Authentic relating, circling, polyamory/ENM community
- **Transformation**: Psychedelic integration, tantra, sacred sexuality
- **Sacred Feminine**: Goddess temples, women's circles, Red Tents, priestess trainings
- **Consciousness**: Meditation sanghas (Vipassana, Zen), mindfulness communities
- **Community**: Ecovillages, intentional communities, land trusts
- **Nature**: Wilderness retreats, forest gatherings
- **Festivals**: Oregon Country Fair, New Culture Summer Camp, tantra festivals

### Features
- **Interactive Filtering**: Filter events by region, type (ecstatic dance, meditation, connection, etc.), and frequency (weekly, monthly, annual, seasonal)
- **AND Logic**: Combine filters to find exactly what you're looking for (e.g., "weekly ecstatic dance in Humboldt")
- **Smart Filters**: Only shows filter options with 2+ events to keep the interface clean
- **Back to Top**: Sticky button for easy navigation on long pages

### Event Indicators
- 🔄 = Recurring (weekly/monthly)
- 🎪 = Workshops/retreats/festivals

### Geographic Coverage
**10 regions** organized north to south for easy trip planning:

**Oregon (3)**: Portland → Eugene → Rogue Valley

**California (7)**: Humboldt → Mt. Shasta → Sonoma → Mendocino → West Marin → SF Bay → Santa Cruz

### Adding Regional Events

1. Edit [regional-events.json](regional-events.json) to add your event:

```json
{
  "name": "🔄 Event Name",
  "url": "https://example.com",
  "schedule": "Wednesdays, 7-10 PM",
  "venue": "Venue Name, City",
  "price": "$15-$20",
  "description": "Optional description with email, newsletter, phone, etc.",
  "tags": ["dance", "weekly"]
}
```

**Tag Taxonomy** (use 1 event type + 1 frequency):
- **Event Types**: `dance`, `connection`, `retreat`, `breathwork`, `consciousness`, `ritual`, `festival`, `music`
- **Frequencies**: `weekly`, `monthly`, `annual`, `seasonal`
- **Emoji Indicators**: 🔄 for recurring (weekly/monthly), 🎪 for workshops/retreats/festivals

2. Validate the data:
```bash
npm run validate-data
```

3. Build the HTML:
```bash
npm run build:regional
```

4. Verify ordering:
```bash
npm run verify-calendar
```

The build script generates [regional-calendar.html](regional-calendar.html) from the JSON data. **Never edit the HTML file directly** - always edit the JSON and rebuild.

## Maintenance Workflows

### Regular Maintenance Tasks

**Monthly** (1st of each month):
1. Verify recurring event schedules haven't changed
2. Check for venue changes or closures
3. Update contact information (emails, phones, newsletters)

**Quarterly** (Jan/Apr/Jul/Oct):
1. **Check which events need date updates**: `npm run check-stale-dates`
2. **Update annual event dates** in [regional-events.json](regional-events.json):
   - Visit event URLs shown by check-stale-dates script
   - Update `schedule` field with confirmed dates, note "(dates TBA for YYYY)" if unconfirmed
   - Always include year in schedule field for annual/seasonal events
3. Validate all external links: `npm run validate-links`
4. Review and prune inactive events (check event URLs for cancellations/closures)
5. Search for new events in underrepresented regions (Lake County, Mendocino)
6. Rebuild after updates: `npm run build:regional`

**Before Every Build**:
1. Validate data quality: `npm run validate-data`
2. Build calendar: `npm run build:regional`
3. Verify region ordering: `npm run verify-calendar`
4. Run tests: `npm test`

### Adding Community Resources

Prefer official websites and direct contact methods (no Facebook/Instagram):
- **Best**: Email newsletters (e.g., "Subscribe: https://example.com/newsletter")
- **Good**: Direct email/phone (e.g., "Email: events@example.org, Phone: 555-1234")
- **Acceptable**: Official websites with event calendars
- **Avoid**: Social media links

### Data Quality Guidelines

1. **Required fields**: All events must have `name`, `url`, `schedule`, `venue`, `price`, `tags`
2. **Tag structure**: Use exactly 1 event type + 1 frequency (e.g., `["dance", "weekly"]`)
3. **Emoji indicators**: Add 🔄 for recurring events, 🎪 for workshops/retreats/festivals
4. **Recurring events**: Use patterns like "Wednesdays, 7-10 PM" (no specific dates)
5. **Annual events**: Estimate dates based on previous years, note in description if unconfirmed
6. **Geographic focus**: Prioritize events within 2-3 hours of Humboldt County

### Data Freshness Strategy

**Static data (doesn't require updates):**
- Recurring event patterns ("Tuesdays, 6-8:30 PM", "2nd Monday")
- Event names, descriptions, tags
- Venue addresses (update only if changed)
- URLs, contact info (update only if changed)

**Dynamic data (requires quarterly updates):**
- **Annual festival dates**: Update every January for summer festivals
  - Check official event websites 3-6 months before event
  - Use "(dates TBA for 2026)" format when dates unconfirmed
  - Always include year in schedule field
- **Seasonal retreat dates**: Update quarterly based on organizer announcements
- **Semi-recurring specific dates**: Events with occasional dates (e.g., "Select Sundays: Dec 28, Jan 11, 25")

**Automation strategy**: Static build with manual quarterly updates is optimal because:
- Client-side fetching would require CORS/API keys (most event sites don't have APIs)
- Event organizers announce dates months in advance (plenty of lead time)
- Quarterly maintenance cycle aligns with festival announcement schedules
- Static patterns (90% of events) never go stale

### Finding New Events

**Search strategies**:
- Google: `"ecstatic dance" OR "5Rhythms" [city name]`
- Google: `"authentic relating" OR "circling" [city name]`
- Google: `"contact improvisation" [city name]`
- Check existing event websites for related communities
- Ask organizers for recommendations

**Geographic priorities** (closest to Humboldt first):
1. Humboldt County, Mendocino County, Lake County
2. Mt. Shasta, Rogue Valley (Ashland)
3. Sonoma County, West Marin
4. Eugene, Santa Cruz
5. SF Bay Area, Portland (endpoints of range)

## Deployment

### GitHub Pages
1. Push to GitHub
2. Settings → Pages → Source: `main` branch, `/` root
3. Site deploys to `https://[username].github.io/coloft/`

### Custom Domain
Add CNAME record pointing to `[username].github.io`

## Technology & Architecture

### Hybrid Static/Dynamic Approach

**Regional Calendar** ([regional-calendar.html](regional-calendar.html)):
- **Static HTML generation**: Build from [regional-events.json](regional-events.json) via `npm run build:regional`
- **Community-sourced data**: Event schedules and details collected from organizer websites - always verify via event links before traveling
- **Recurring patterns**: Events show schedules like "Tuesdays, 6-8:30 PM" (no rebuild needed)
- **One-off events**: Annual festivals/retreats have estimated dates based on previous years - check official sites for confirmation
- **Client-side filtering**: JavaScript-powered AND-logic filtering works without page reload
- **Progressive enhancement**: Core content works without JavaScript

**Local Events** ([index.html](index.html)):
- **3 active events starting mid-January 2026**:
  - Somatic Lab Loft Sessions: Select Sundays at 6:00 PM (starts Jan 25, 2026)
  - Authentic Connection Circle: Every Tuesday at 5:30 PM (starts Jan 13, 2026)
  - Relating Games: Every other Saturday at 3:00 PM (starts Jan 24, 2026)
- **Static HTML** with embedded event metadata (manually maintained)
- **Dynamic date calculation**: [scripts/dates.js](scripts/dates.js) runs in browser on page load
  - Hardcoded recurrence rules for each event (select Sundays, every Tuesday, etc.)
  - Each event has its own start date (Somatic Lab Loft Sessions: Jan 25, Authentic Connection Circle: Jan 13)
  - Calculates "Next 3 occurrences" from today's date (always current)
  - Updates `.event-next` elements automatically
  - **Easy schedule changes**: Edit `EVENT_SCHEDULES` at top of scripts/dates.js - no rebuild needed!
- **Always fresh**: Dates never go stale - calculated from today on every page load
- **Print-optimized**: Individual event pages fit perfectly on 1 page (US Letter)
- **Event metadata**: [events.json](events.json) stores event configuration (descriptions, themes, etc.)
  - Currently **not used** by date calculation (reserved for future data-driven architecture)
  - Event metadata currently duplicated in HTML files

### Stack
- **Pure HTML/CSS/JS**: No frameworks, minimal JavaScript
- **Static site**: Fast, secure, free hosting on GitHub Pages
- **Build step**: Simple Node.js script generates regional calendar HTML from JSON
- **Automated testing**: Puppeteer-core for print validation
- **Progressive enhancement**: Works without JavaScript

## Contact

**events@coloft.org**

---

*Built with ❤️ for the healing collective community*
