import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

const _storageKey = 'device_id';

String? _cachedDeviceId;

Future<String> getOrCreateDeviceId() async {
  if (_cachedDeviceId != null) return _cachedDeviceId!;

  final prefs = await SharedPreferences.getInstance();
  var id = prefs.getString(_storageKey);
  if (id == null || id.isEmpty) {
    id = const Uuid().v4();
    await prefs.setString(_storageKey, id);
  }
  _cachedDeviceId = id;
  return id;
}
