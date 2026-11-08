; Inno Setup script. Build after PyInstaller:
;   iscc /DAppVersion=0.0.1 packaging\windows\installer.iss
#ifndef AppVersion
  #define AppVersion "0.0.1"
#endif

[Setup]
AppId={{6C0B3F0E-5D7A-4B7E-9C51-2F1E5A9C0D11}
AppName=SpaceRockSuite
AppVersion={#AppVersion}
AppPublisher=Ella Agnes Puerto
AppPublisherURL=https://github.com/eagnespuerto/spacerocksuite
AppSupportURL=https://github.com/eagnespuerto/spacerocksuite/issues
AppUpdatesURL=https://github.com/eagnespuerto/spacerocksuite/releases
DefaultDirName={localappdata}\Programs\SpaceRockSuite
DefaultGroupName=SpaceRockSuite
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\..\dist
OutputBaseFilename=SpaceRockSuite-{#AppVersion}-setup
SetupIconFile=..\icons\spacerocksuite.ico
UninstallDisplayIcon={app}\SpaceRockSuite.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
LicenseFile=..\..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "Add the srs command to PATH"; Flags: unchecked

[Files]
Source: "..\..\dist\SpaceRockSuite\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SpaceRockSuite"; Filename: "{app}\SpaceRockSuite.exe"
Name: "{group}\Uninstall SpaceRockSuite"; Filename: "{uninstallexe}"
Name: "{autodesktop}\SpaceRockSuite"; Filename: "{app}\SpaceRockSuite.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\SpaceRockSuite.exe"; Description: "{cm:LaunchProgram,SpaceRockSuite}"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Param) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;
