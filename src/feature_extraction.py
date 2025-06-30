from collections import defaultdict
import yaml
from pathlib import Path

def load_yml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_geom_data(geom_list):
    objects = defaultdict(list)
    for item in geom_list:
        if 'geom' in item:
            g = item['geom']
            obj_id = g['id1']
            bbox = list(map(int, g['g0'].split()))
            ts = g['ts0']
            objects[obj_id].append({'frame': ts, 'bbox': bbox})
    return objects

def parse_activity_data(activity_list):
    events = []
    for item in activity_list:
        if 'act' in item:
            a = item['act']
            event_type = list(a['act2'].keys())[0]
            event_id = a['id2']
            ts_start, ts_end = a['timespan'][0]['tsr0']
            actors = [actor['id1'] for actor in a.get('actors', [])]
            events.append({
                'event_id': event_id,
                'type': event_type,
                'start': ts_start,
                'end': ts_end,
                'duration': ts_end - ts_start,
                'actors': actors
            })
    return events

def parse_region_data(region_list):
    regions = defaultdict(list)
    for item in region_list:
        if 'regions' in item:
            r = item['regions']
            ts = r['ts0']
            poly = r['poly0']
            regions[ts].append(poly)
    return regions

def extract_event_features(events, geom_data, region_data=None):
    features = []
    for e in events:
        actor_counts = len(e['actors'])
        duration = e['duration']
        avg_width = []
        avg_height = []

        for obj_id in e['actors']:
            track = geom_data.get(obj_id, [])
            if track:
                bboxes = [b['bbox'] for b in track if e['start'] <= b['frame'] <= e['end']]
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    avg_width.append(x2 - x1)
                    avg_height.append(y2 - y1)

        w = sum(avg_width) / len(avg_width) if avg_width else 0
        h = sum(avg_height) / len(avg_height) if avg_height else 0

        num_regions_in_event = 0
        if region_data:
            for f in range(e['start'], e['end'] + 1):
                if f in region_data:
                    num_regions_in_event += len(region_data[f])

        features.append({
            'type': e['type'],
            'duration': duration,
            'num_actors': actor_counts,
            'avg_width': w,
            'avg_height': h,
            'regions_in_event': num_regions_in_event
        })
    return features

def extract_features_from_video(activities_path, geom_path, regions_path=None):
    activities_raw = load_yml(activities_path)
    geom_raw = load_yml(geom_path)
    region_raw = load_yml(regions_path) if regions_path else None

    events = parse_activity_data(activities_raw)
    geom_data = parse_geom_data(geom_raw)
    region_data = parse_region_data(region_raw) if region_raw else None

    features = extract_event_features(events, geom_data, region_data)
    return features

if __name__ == "__main__":
    features = extract_features_from_video(
        "viratannotations/train/VIRAT_S_000001.activities.yml",
        "viratannotations/train/VIRAT_S_000001.geom.yml",
        "viratannotations/train/VIRAT_S_000001.regions.yml"
    )

    from pprint import pprint
    pprint(features[:3])  # mostrar primeiros eventos com features
