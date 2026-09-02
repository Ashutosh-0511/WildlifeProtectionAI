from ml.events.pipeline import EventOrchestrator


def main():
    orchestrator = EventOrchestrator()
    event = orchestrator.build_event(
        camera_id="CAM-001",
        zone_id="ZONE-001",
        track_id="1",
        species="elephant",
        behaviour="AGGRESSIVE_ABNORMAL",
        human_present=True,
        distance_m=4.0,
        detector_confidence=0.95,
        behaviour_confidence=0.90,
        persistence_s=20,
        detector_version="MDV6-yolov9-c",
        tracker_version="ByteTrack",
        behaviour_model_version="resnet18_lstm",
    )
    assert event.risk_level in {"HIGH", "CRITICAL"}
    assert len(event.factors) >= 7
    assert orchestrator.should_alert(event)
    assert not orchestrator.should_alert(event)
    print(event.to_dict())
    print("RISK_ENGINE_OK")
    print("ALERT_DEDUP_OK")


if __name__ == "__main__":
    main()
