import requests
import json
from msal import ConfidentialClientApplication
from urllib.parse import quote


class EmployeeHRDB:
    """
    A client for fetching employee records, organization structures, and manager
    information from Dataverse, designed to provide a compact user profile for an agent.
    """

    def __init__(self, org_url: str, tenant_id: str, client_id: str, client_secret: str):
        """
        Initialize the Dataverse client with authentication details.

        Args:
            org_url: Dataverse organization URL (e.g., 'https://yourorg.crm.dynamics.com')
            tenant_id: Azure AD tenant ID
            client_id: Azure AD application (client) ID
            client_secret: Azure AD client secret
        """
        self.org_url = org_url.rstrip('/')
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.headers = None

        # Acquire token on initialization
        self._acquire_token()

    def _acquire_token(self) -> None:
        """Acquire an access token using client credentials and set up request headers."""
        app = ConfidentialClientApplication(
            self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
            client_credential=self.client_secret
        )
        result = app.acquire_token_for_client(
            scopes=[f"{self.org_url}/.default"])

        if "access_token" in result:
            self.access_token = result["access_token"]
            self.headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "OData-MaxVersion": "4.0",
                "OData-Version": "4.0"
            }
            print("✅ Successfully acquired access token.")
        else:
            raise RuntimeError(
                f"Could not acquire token: {result.get('error_description')}")

    def _request(self, url: str, method: str = "GET", **kwargs) -> dict:
        """
        Perform a Dataverse API request with automatic token refresh (simple implementation).

        Args:
            url: Full API URL
            method: HTTP method (default 'GET')
            **kwargs: Additional arguments passed to requests.request

        Returns:
            JSON response as dict

        Raises:
            Exception on non-200 status.
        """
        if not self.headers:
            self._acquire_token()

        response = requests.request(
            method, url, headers=self.headers, **kwargs)

        # If unauthorized, refresh token and retry once
        if response.status_code == 401:
            self._acquire_token()
            response = requests.request(
                method, url, headers=self.headers, **kwargs)

        if response.status_code != 200:
            print(
                f"❌ Request failed: {response.status_code} - {response.text}")
            response.raise_for_status()

        return response.json()

    def get_org_structure_record(self, org_guid: str) -> dict | None:
        """
        Fetch the full JSON record for a specific Organization Structure using its GUID.

        Args:
            org_guid: GUID of the organization structure (with or without curly braces)

        Returns:
            Organization structure record as dict, or None if not found.
        """
        clean_id = org_guid.strip('{}')
        url = f"{self.org_url}/api/data/v9.1/cr603_organizationstructures({clean_id})"
        try:
            return self._request(url)
        except Exception as e:
            print(f"Error fetching organization structure: {e}")
            return None

    def get_user_with_org_display_name(self, email: str, table_name: str = "hr_employees",
                                       display_name_fields: list[str] = None) -> dict | None:
        """
        Fetch a user record and retrieve its organization's display name.

        Args:
            email: User email (hr_useremail)
            table_name: Dataverse logical name of the employee table (default 'hr_employees')
            display_name_fields: List of candidate field names to use for the display name.

        Returns:
            User record dict with added 'org_display_name' and 'organization' keys.
        """
        if display_name_fields is None:
            display_name_fields = [
                'cr603_name', 'cr603_displayname', 'cr603_fullname',
                'cr603_organizationname', 'cr603_legalname', 'name'
            ]

        # Fetch user record
        filter_query = f"$filter=hr_useremail eq '{email}'"
        url = f"{self.org_url}/api/data/v9.1/{table_name}?{filter_query}"
        data = self._request(url)
        users = data.get('value', [])
        if not users:
            print(f"No user found with email: {email}")
            return None

        user_record = users[0]

        # Fetch user's organization (position) record
        org_guid = user_record.get('_cr603_organizationstructure_value')
        if org_guid:
            org_data = self.get_org_structure_record(org_guid)
            if org_data:
                user_record['organization'] = org_data
                # Find display name
                display_name = None
                for field in display_name_fields:
                    val = org_data.get(field)
                    if val and str(val).strip():
                        display_name = str(val).strip()
                        break
                user_record['org_display_name'] = display_name
                if display_name:
                    print(f"✅ User's org display name: {display_name}")
                else:
                    print("⚠️ No display name found for user's organization.")
            else:
                user_record['organization'] = None
                user_record['org_display_name'] = None
        else:
            user_record['organization'] = None
            user_record['org_display_name'] = None
            print("No organization linked to this user.")

        return user_record

    def get_user_with_org_manager(self, email: str, table_name: str = "hr_employees",
                                  display_name_fields: list[str] = None,
                                  fetch_manager: bool = True) -> dict | None:
        """
        Fetch a user record, its organization, and optionally the manager's organization record.

        Args:
            email: User email (hr_useremail)
            table_name: Employee table logical name
            display_name_fields: List of candidate fields for display names
            fetch_manager: Whether to fetch the manager's organization record

        Returns:
            User record dict with additional keys: 'organization', 'org_display_name',
            'manager_organization', 'manager_org_display_name'.
        """
        if display_name_fields is None:
            display_name_fields = [
                'cr603_name', 'cr603_displayname', 'cr603_fullname',
                'cr603_organizationname', 'cr603_legalname', 'name'
            ]

        # 1) Get basic user + organization
        user_record = self.get_user_with_org_display_name(
            email, table_name, display_name_fields)
        if not user_record:
            return None

        # 2) Fetch manager organization if requested and user has an organization
        if fetch_manager and user_record.get('organization'):
            org_data = user_record['organization']
            manager_guid = org_data.get('_cr603_administrativemanager_value') or org_data.get(
                '_cr603_technicalmanager_value')

            if manager_guid:
                mgr_data = self.get_org_structure_record(manager_guid)
                if mgr_data:
                    user_record['manager_organization'] = mgr_data
                    mgr_display = None
                    for field in display_name_fields:
                        val = mgr_data.get(field)
                        if val and str(val).strip():
                            mgr_display = str(val).strip()
                            break
                    user_record['manager_org_display_name'] = mgr_display
                    print(f"✅ Manager's org display name: {mgr_display}")
                else:
                    user_record['manager_organization'] = None
                    user_record['manager_org_display_name'] = None
            else:
                print("No manager GUID found in user's organization record.")
                user_record['manager_organization'] = None
                user_record['manager_org_display_name'] = None
        else:
            user_record['manager_organization'] = None
            user_record['manager_org_display_name'] = None
            if not fetch_manager:
                print("Manager fetch skipped.")

        return user_record

    def build_agent_user_profile(self, user_record: dict) -> dict:
        """
        Transform the full user record (from get_user_with_org_manager) into a compact
        payload suitable for the OPDHealthcareAgent.

        Args:
            user_record: The dictionary returned by get_user_with_org_manager or similar.

        Returns:
            Compact profile dict with essential fields.
        """
        org = user_record.get('organization', {})
        mgr = user_record.get('manager_organization', {})

        # Determine BU: prefer hr_userbu_fx, fallback to function category or department
        bu = user_record.get('hr_userbu_fx') or user_record.get(
            'hr_functioncategory') or user_record.get('hr_department')

        # Determine doctor_name: only if position contains "Doctor"
        position = user_record.get('hr_position', '')
        doctor_name = user_record.get(
            'hr_fullname') if "Doctor" in position else None

        profile = {
            # Basic identity
            "hr_fullname": user_record.get('hr_fullname'),
            "hr_firstname": user_record.get('hr_firstname'),
            "hr_email": user_record.get('hr_email'),
            "hr_position": user_record.get('hr_position'),
            "hr_adtitle": user_record.get('hr_adtitle'),
            "hr_department": user_record.get('hr_department'),
            "hr_functioncategory": user_record.get('hr_functioncategory'),
            "hr_section": user_record.get('hr_section'),
            "hr_jobtype": user_record.get('hr_jobtype'),
            "hr_employeestatus": user_record.get('hr_employeestatus'),
            "hr_hiredate": user_record.get('hr_hiredate'),

            # Organisation / position
            "org_display_name": user_record.get('org_display_name'),
            "org_structure_id": org.get('hr_organizationstructureid'),

            # Business unit (critical for data filtering)
            "bu": bu,

            # Doctor flag (if applicable)
            "doctor_name": doctor_name,

            # Manager info (if manager exists)
            "manager": {
                "name": mgr.get('hr_fullnameofcurrentemployee'),
                "position": mgr.get('cr603_name'),
                "department": mgr.get('hr_department'),
                "employee_guid": mgr.get('_hr_currentemployee_value'),
                "org_structure_id": mgr.get('hr_organizationstructureid')
            } if mgr else None,

            # Compatibility with existing agent code that expects a top-level manager name
            "hr_administrativemanagername": mgr.get('hr_fullnameofcurrentemployee') if mgr else None,

            # Fallback job title (agent uses hr_jobtitle for tone)
            "hr_jobtitle": user_record.get('hr_position') or user_record.get('hr_adtitle')
        }
        return profile

    def get_user_profile(self, email: str, table_name: str = "hr_employees") -> dict | None:
        """
        Convenience method that returns the agent-ready compact user profile for a given email.

        Args:
            email: User email (hr_useremail)
            table_name: Employee table logical name

        Returns:
            Compact profile dict (as per build_agent_user_profile) or None if user not found.
        """
        full_record = self.get_user_with_org_manager(
            email, table_name, fetch_manager=True)
        if not full_record:
            return None
        return self.build_agent_user_profile(full_record)

    def get_user_by_position_and_bu(self, position_name: str, bu_code: str) -> dict | None:
        # Try plural first (most common), then singular
        for entity_set in ["hr_employees", "hr_employee"]:
            safe_position = quote(position_name)
            filter_query = f"$filter=hr_adtitle eq '{safe_position}' or hr_position eq '{safe_position}' or hr_positionname eq '{safe_position}' and hr_userbu_fx eq '{bu_code}'&$top=1"
            api_url = f"{self.org_url}/api/data/v9.1/{entity_set}?{filter_query}"
            try:
                data = self._request(api_url)
                users = data.get('value', [])
                if users:
                    print(f"Found using entity set: {entity_set}")
                    employee_record = users[0]
                    # Enrich with org display name
                    email = employee_record.get('hr_useremail')
                    if email:
                        enriched = self.get_user_with_org_display_name(email, entity_set)
                        if enriched:
                            return enriched
                    return employee_record
            except Exception as e:
                print(f"Failed to fetch by position from {entity_set}: {e}")
        print("No employee found with any entity set")
        return None



# Example usage (if run as a script)
if __name__ == "__main__":
    # Replace with your actual Dataverse connection details
    DATAVERSE_ORG_URL = "https://org2f45e702.crm4.dynamics.com"
    DATAVERSE_TENANT_ID = "c515f6b1-812f-4d6c-9542-d914e95b3df1"
    DATAVERSE_CLIENT_ID = "72cf461b-f9d2-4c88-a009-52abef2db39c"
    DATAVERSE_CLIENT_SECRET = "fme8Q~I5JVCfGoxFktYoKonrVsH4vYfmxvPAZa7J"

    hr_db = EmployeeHRDB(
        org_url=DATAVERSE_ORG_URL,
        tenant_id=DATAVERSE_TENANT_ID,
        client_id=DATAVERSE_CLIENT_ID,
        client_secret=DATAVERSE_CLIENT_SECRET
    )

    # Get a user's compact profile
    user_profile = hr_db.get_user_profile("Youssef.MohiElDin@Andalusiagroup.net")
    if user_profile:
        import json
        print(json.dumps(user_profile, indent=2, default=str))
