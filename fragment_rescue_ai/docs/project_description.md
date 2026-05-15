# Fragment Rescue AI — Hackathon Project Description

## Українською

**Fragment Rescue AI** — це система комп’ютерного зору для дрона, яка допомагає рятувальникам знаходити людей у зоні катастрофи навіть тоді, коли постраждалого видно лише частково. Програма аналізує відео з дрона, шукає повну людину або фрагменти тіла, визначає рух, оцінює рівень небезпеки та пояснює, чому саме цю зону потрібно перевірити першою.

Після катастрофи людину часто не видно повністю: вона може бути за уламками, під ковдрою, за коробкою, за стільцем, у темному куті або частково перекрита. Звичайний детектор людей може не спрацювати, бо не бачить повного силуету. Fragment Rescue AI змінює підхід: система шукає не тільки людину цілком, а й **ознаки людини** — руку, ногу, голову, частину тулуба, ключові точки тіла або підозрілий рух.

MVP використовує YOLO для виявлення людей, YOLO pose estimation для пошуку ключових точок тіла, OpenCV для аналізу руху та евристичну логіку для визначення пріоритету перевірки. Якщо система бачить фрагмент людини й рух, вона підвищує рівень небезпеки та повідомляє рятувальникам: **“Check this area first.”**

Основна цінність проєкту — не просто показати рамку навколо людини, а пояснити рішення: які ознаки знайдено, чому є підозра на прихованого постраждалого і який рівень пріоритету має ця зона.

## English

**Fragment Rescue AI** is a drone-based computer vision system for disaster response. It detects not only fully visible people, but also partially hidden human signs such as arms, legs, heads, body parts, silhouettes, and small movements. The system analyzes the drone video stream using YOLO, pose estimation, and OpenCV, assigns a threat level, calculates confidence, and explains why a certain area should be checked first by rescuers.

After a disaster, victims are often not fully visible. A person may be behind debris, under a blanket, behind a box, behind a chair, in a dark corner, or partially occluded by the environment. A standard person detector may fail because it expects a clear full-body view. Fragment Rescue AI focuses on **human evidence**, not only full-body detection.

The MVP combines YOLO person detection, YOLO pose keypoint analysis, OpenCV motion detection, and interpretable rescue logic. When the system detects a possible human fragment together with movement, it increases the threat level and recommends a high-priority rescue check.

The key idea is explainability: the system does not only say “person detected” or “not detected.” It says: **“I don’t see a full person, but I see human signs. This area should be checked first.”**

## 30–60 second pitch in English

Our project, **Fragment Rescue AI**, helps rescuers find people who are not fully visible after a disaster. In real rescue scenes, a victim may be hidden behind debris, furniture, smoke, blankets, or darkness, so a normal person detector can miss them. Our system analyzes drone video and searches not only for a full human body, but also for partial human signs: wrists, elbows, shoulders, head, legs, pose keypoints, silhouettes, and small movements. It combines YOLO, pose estimation, and OpenCV motion detection to calculate a human probability score and assign a threat level from green to red. Most importantly, it explains its decision, so rescuers can understand why a certain area should be checked first. The goal is simple: faster search, better prioritization, and a higher chance of finding hidden survivors.

## Український pitch на 30–60 секунд

Наш проєкт **Fragment Rescue AI** допомагає рятувальникам знаходити людей, яких після катастрофи видно не повністю. У реальних умовах постраждалий може бути за уламками, під пледом, за меблями, у темному куті або частково перекритий, тому звичайний детектор людей може його не помітити. Наша система аналізує відео з дрона і шукає не тільки повну людину, а й часткові ознаки: руку, ногу, голову, плечі, лікті, ключові точки тіла, силует або невеликий рух. Ми поєднуємо YOLO, pose estimation та OpenCV motion detection, щоб визначити ймовірність присутності людини, рівень небезпеки та рекомендацію для рятувальників. Головна ідея: якщо система не бачить повну людину, але бачить людські ознаки, вона підказує — цю зону потрібно перевірити першою.

## Future improvements

- Thermal camera support;
- GPS mapping;
- Automatic area scanning;
- Rescue priority map;
- Cloud dashboard;
- Offline mode;
- Voice alerts;
- Integration with real drone SDK;
- Segmentation model;
- Custom dataset training for disaster scenes;
- Body-part detector for occluded people;
- Multi-camera fusion;
- Edge deployment on Jetson or Raspberry Pi.
