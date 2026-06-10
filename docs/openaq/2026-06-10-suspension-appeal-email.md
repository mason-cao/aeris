# Email draft — OpenAQ suspension appeal

> Draft, 2026-06-10. To: dev@openaq.org. Send from the account email
> (masoncao7@gmail.com). Before sending: confirm only one account/key was ever
> registered, confirm the Acer's scheduled collector no longer hits OpenAQ, and
> fill in the date the log shows requests started failing if it differs.

---

**Subject:** API suspension — likely caused by my backfill script (account: masoncao7@gmail.com)

Hi,

My API access was suspended for a Terms of Use violation, and after going back through my own code I'm fairly sure I caused it. I want to explain what happened, what I've already changed, and ask what I'd need to do to get access restored.

I'm a high school student building an air-quality research project under the mentorship of a Georgia Tech atmospheric science professor. It monitors a 50 km radius around Houston, cross-references OpenAQ ground measurements with Sentinel-5P, NOAA GFS, and surface weather data, and evaluates how well a small locally-hosted language model can explain detected anomalies. It's non-commercial, and OpenAQ is credited as a data source.

What I believe tripped the suspension:

1. A historical backfill script I ran recently, filling in data back to May 1, paginated measurements sensor-by-sensor across every Houston-area location with only a 100 ms pause between sensors and no pause between pages. I wrote it without checking the documented rate limits, and it would have exceeded them by a wide margin.
2. My hourly collector also re-fetched the full location and sensor list on every run with unthrottled sequential requests instead of caching it.

That was carelessness on my part, not an attempt to scrape or over-consume the platform. I have one account and one API key.

What I've already done:

- Stopped the scheduled collector as soon as I saw the suspension notice.
- Rewrote the backfill so all historical and bulk fetching comes from the public S3 data archive instead of the hosted API, which now only ever serves a light hourly poll of latest values.
- Added a global rate limiter that keeps every API request well under the documented limit, honors Retry-After on 429 responses, and caches the location list daily instead of refetching it every hour.

Is there anything else you'd want changed before restoring access? I'm happy to keep to whatever request budget you consider reasonable. The hourly poll over the Houston-area locations is all I need from the hosted API going forward.

Sorry for the trouble, and thank you for maintaining the platform. It's the backbone of my project.

Mason Cao
masoncao7@gmail.com
