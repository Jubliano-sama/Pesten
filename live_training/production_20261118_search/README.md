# Learned search for the Pesten web player

The release went live on 18 November 2026 at 18:55 UTC from code revision `24d0d0f`. The public home page returned HTTP 200 and public health reported the frozen search profile. A live WebSocket game delivered a private hand, acknowledged two human moves and returned an AI reply. The test room was removed. Origin checks verified all 19 application and model files, a healthy container with zero restarts and the retained rollback container. [Public HTTP verification](public_http.json) · [Live game](public_socket.json) · [Origin verification](final_verification.json)

The web player uses the frozen November search profile in rooms that start with two, three or four players. Each AI seat maintains its own 24-view recurrent state and public discard history. Rooms that start larger keep their previous player selection throughout the game, including after someone finishes or leaves. The web engine continues to handle immediate wins, suit declarations and post-draw restrictions. [Benchmark and fit instructions](../improvement_20261118/RESULTS.md) · [Frozen profile](../improvement_20261118/card_belief/profile.json)

The image builds on the September CPU serving image. It uses the original portable teacher, the evaluated rollout and hand models, and the same Numba and llvmlite versions as the benchmark. The build verifies the profile and compiles the simulation kernels. The HTTP health response identifies the profile, its SHA-256, table sizes, rollout count and horizon. [Dockerfile](../../Dockerfile.web) · [Application and model hashes](image_files.json) · [Previous public health](previous_health.json)

The staging image replayed 640 decisions from 36 fresh native games. Every legal move selected by the serving runtime matched the frozen finalist, including 48 moves where search changed the teacher's choice. The largest absolute difference between legal logits was 0.00000382. Eight complete web-engine games exercised 2,523 decisions at two, three, four, five and eight players. [Cross-runtime replay](runtime_parity.json) · [Web-game evidence](staging_games.json)

The release switch verifies the staged image, all 19 application and model file hashes, the previous container and the Pesten Compose entry. It keeps the previous container with its model and settings for rollback. The origin stores the private journal and original configuration in the release directory. To roll back this release on the origin, run the following command. [Switch implementation](deploy_verified.py)

```sh
python3 /root/pesten-deployments/20261118-search/deploy_verified.py rollback
```
